use flexi_logger::{FileSpec, Logger, WriteMode};
use log::debug;
use std::cell::UnsafeCell;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use io_uring::{opcode, types, IoUring};


const BATCH_SIZE: usize = 1024 * 1024; // 1 MB — also the address-slot width
const RING_LEN:   usize = 4;


/// Aligned, interior-mutable byte buffer.
pub struct Buffer {
    /// Raw pointer to the aligned allocation, wrapped in [`UnsafeCell`] to
    /// allow interior mutability without a lock.
    pub(crate) buffer: UnsafeCell<*mut u8>,
    /// Total allocation size in bytes.
    size: AtomicUsize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(capacity: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(capacity, 4096)
            .expect("invalid layout");
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        Self {
            buffer: UnsafeCell::new(ptr),
            size: AtomicUsize::new(capacity),
        }
    }

    /// Copy `payload` into the buffer at `offset`.
    /// Caller must guarantee `offset + payload.len() <= capacity`.
    pub fn write(&self, offset: usize, payload: &[u8]) {
        unsafe {
            let dst = (*self.buffer.get()).add(offset);
            std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
        }
    }

    /// Raw pointer `offset` bytes in — handed directly to io_uring.
    pub fn as_ptr(&self, offset: usize) -> *const u8 {
        unsafe { (*self.buffer.get()).add(offset) }
    }

    pub fn capacity(&self) -> usize {
        self.size.load(AtomicOrdering::Relaxed)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let cap = self.size.load(AtomicOrdering::Relaxed);
        let layout = std::alloc::Layout::from_size_align(cap, 4096)
            .expect("invalid layout");
        unsafe { std::alloc::dealloc(*self.buffer.get(), layout) };
    }
}

// ── Slot ─────────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum SlotState {
    Open,
    InFlight,
}

struct Slot {
    buf: Buffer,
    state: SlotState,

    /// Disk offset assigned at seal time via `fetch_add(valid_len)`.
    /// Zero until the slot is submitted.
    disk_offset: u64,

    /// How many bytes were actually written before sealing (≤ BATCH_SIZE).
    sealed_len: usize,
}

impl Slot {
    fn new() -> Self {
        Self {
            buf:         Buffer::new(BATCH_SIZE),
            state:       SlotState::Open,
            disk_offset: 0,
            sealed_len:  0,
        }
    }
}

// ── CompletedRange ────────────────────────────────────────────────────────────

/// A flushed address range: the reserved disk start and actual byte count.
#[derive(Debug, Clone, Copy)]
pub struct CompletedRange {
    /// Start of the 1 MB address slot on disk.
    pub disk_offset: u64,
    /// Bytes actually written within that slot (≤ BATCH_SIZE).
    pub sealed_len:  usize,
}


/// Rotates through `RING_LEN` pre-allocated 1 MB slots.
///
/// Each slot is assigned a disk address range at **seal time**, sized to
/// exactly the bytes it contains (`valid_len`), so `file_offset` advances by
/// the true write size rather than a padded block.  Slots may be arbitrarily
/// underfilled.
///
/// A slot is sealed (submitted as one io_uring `Write`) when:
/// - an incoming payload would overflow it  →  automatic, via `append`
/// - the caller calls `flush()`             →  explicit mid-stream seal
/// - the caller calls `finish()`            →  final seal + drain
///
/// On completion each CQE produces a [`CompletedRange`].  `finish()` returns
/// all ranges accumulated so the caller can sort and read them back.
struct RingWriter {
    slots:       [Slot; RING_LEN],
    /// Index of the slot currently accepting writes.
    head:        usize,
    /// Write cursor within `slots[head]`; claimed via `fetch_add`.
    cursor:      AtomicUsize,
    /// Shared counter advanced by `valid_len` at seal time — never padded.
    file_offset: Arc<AtomicU64>,
    ring:        IoUring,
    pending:     usize,
    fd:          i32,
    /// Ranges collected from completed CQEs.
    completed:   Vec<CompletedRange>,
}

impl RingWriter {
    fn new(
        fd: i32,
        file_offset: Arc<AtomicU64>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            slots:       std::array::from_fn(|_| Slot::new()),
            head:        0,
            cursor:      AtomicUsize::new(0),
            file_offset,
            ring:        IoUring::new(256)?,
            pending:     0,
            fd,
            completed:   Vec::new(),
        })
    }

    // ── public API ────────────────────────────────────────────────────────────

    pub fn append(
        &mut self,
        payload: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let len = payload.len();

        // Claim a unique, non-overlapping range in the current slot.
        let offset = self.cursor.fetch_add(len, AtomicOrdering::SeqCst);

        if offset + len > BATCH_SIZE {
            // Payload won't fit: seal whatever was written before our claim,
            // rotate to a fresh slot, then write there.
            // The slot is allowed to be underfilled — no minimum occupancy.
            self.seal_and_advance(offset)?;

            let new_offset = self.cursor.fetch_add(len, AtomicOrdering::SeqCst);
            self.slots[self.head].buf.write(new_offset, payload);
        } else {
            // Fits: write and leave the slot open for more data.
            // Sealing happens only on the next overflow or at finish().
            self.slots[self.head].buf.write(offset, payload);
        }

        Ok(())
    }

    /// Seal the current slot mid-stream and rotate to the next one.
    /// The writer remains usable — `append` continues into the fresh slot.
    /// Completed ranges are accumulated internally and returned by `finish()`.
    pub fn flush(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let filled = self.cursor.load(AtomicOrdering::SeqCst);
        if filled > 0 {
            self.seal_and_advance(filled)?;
        }
        Ok(())
    }

    /// Seal the partial tail, drain all in-flight writes, and return every
    /// completed range so the caller can sort + read them back.
    pub fn finish(
        &mut self,
    ) -> Result<Vec<CompletedRange>, Box<dyn std::error::Error + Send + Sync>> {
        let remaining = self.cursor.load(AtomicOrdering::SeqCst);
        if remaining > 0 {
            self.seal_and_advance(remaining)?;
        }

        // Wait for every outstanding SQE to complete.
        if self.pending > 0 {
            self.ring.submit_and_wait(self.pending)?;
            self.reap_all();
        }

        Ok(std::mem::take(&mut self.completed))
    }

    // ── internals ─────────────────────────────────────────────────────────────

    /// Seal `slots[head]` with `valid_len` actual bytes, submit the write,
    /// then rotate `head` — blocking if the next slot is still InFlight.
    fn seal_and_advance(
        &mut self,
        valid_len: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if valid_len > 0 {
            self.submit_slot(self.head, valid_len)?;
        }

        let next = (self.head + 1) % RING_LEN;

        // If the next slot is still InFlight the ring is full — wait.
        while self.slots[next].state == SlotState::InFlight {
            self.ring.submit_and_wait(1)?;
            self.reap_all();
        }

        self.head = next;
        self.cursor.store(0, AtomicOrdering::SeqCst);

        Ok(())
    }

    /// Submit one io_uring `Write` for `slots[idx]`.
    ///
    /// The file offset is claimed here at seal time using `valid_len`, so the
    /// global counter advances by exactly the bytes actually written —
    /// never a padded or fixed block size.
    fn submit_slot(
        &mut self,
        idx: usize,
        valid_len: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Claim exactly `valid_len` bytes in the file at seal time.
        let disk_off = self
            .file_offset
            .fetch_add(valid_len as u64, AtomicOrdering::SeqCst);

        self.slots[idx].disk_offset = disk_off;
        self.slots[idx].sealed_len  = valid_len;
        self.slots[idx].state       = SlotState::InFlight;

        // Pack both slot index (high 32 bits) and sealed_len (low 32 bits)
        // into user_data so the CQE handler has everything it needs.
        let user_data = ((idx as u64) << 32) | (valid_len as u64);

        let write_e = opcode::Write::new(
            types::Fd(self.fd),
            self.slots[idx].buf.as_ptr(0),
            valid_len as u32,
        )
        .offset(disk_off)
        .build()
        .user_data(user_data);

        unsafe {
            if self.ring.submission().push(&write_e).is_err() {
                self.ring.submit()?;
                self.ring
                    .submission()
                    .push(&write_e)
                    .expect("submission queue full after drain");
            }
        }

        self.pending += 1;
        Ok(())
    }

    /// Drain every available CQE, free its slot, and record the completed range.
    fn reap_all(&mut self) {
        while let Some(cqe) = self.ring.completion().next() {
            let ud  = cqe.user_data();
            let idx = (ud >> 32) as usize;
            let len = (ud & 0xFFFF_FFFF) as usize;

            self.completed.push(CompletedRange {
                disk_offset: self.slots[idx].disk_offset,
                sealed_len:  len,
            });

            self.slots[idx].state      = SlotState::Open;
            self.slots[idx].sealed_len = 0;
            self.pending -= 1;
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _logger = Logger::try_with_str("debug")
        .unwrap()
        .log_to_file(
            FileSpec::default()
                .directory("logs")
                .basename("verification")
                .suffix("log"),
        )
        .write_mode(WriteMode::BufferAndFlush)
        .start()
        .unwrap();

    println!("Writing 150,000 messages using io_uring (4-slot 1 MB ring, pre-reserved addresses)...");

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .truncate(true)
        .open("data_output.log")?;

    let fd          = file.as_raw_fd();
    let file_offset = Arc::new(AtomicU64::new(0));

    let start       = Instant::now();
    let mut handles = vec![];

    for thread_id in 0..5usize {
        let file_offset_clone = Arc::clone(&file_offset);

        let handle = thread::spawn(
            move || -> Result<Vec<CompletedRange>, Box<dyn std::error::Error + Send + Sync>> {
                let mut writer = RingWriter::new(fd, file_offset_clone)?;

                for i in 0..30_000usize {
                    let message_num = i + 30_000 * thread_id;
                    let message = format!(
                        "[{:02}] [Thread-{}] Message {:06}\n",
                        thread_id, thread_id, message_num
                    );
                    writer.append(message.as_bytes())?;
                }

                let ranges = writer.finish()?;
                println!(
                    "Thread {} done — {} completed range(s)",
                    thread_id,
                    ranges.len()
                );
                Ok(ranges)
            },
        );
        handles.push(handle);
    }

    // Collect every range from every thread.
    let mut all_ranges: Vec<CompletedRange> = Vec::new();
    let mut total_messages = 0usize;

    for handle in handles {
        let ranges = handle.join().unwrap().expect("thread panicked");
        total_messages += 30_000; // each thread writes exactly 30,000 messages
        all_ranges.extend(ranges);
    }

    drop(file);

    let elapsed = start.elapsed();
    println!(
        "Writing complete! {} messages in {:.2}s ({:.0}/s)",
        total_messages,
        elapsed.as_secs_f64(),
        total_messages as f64 / elapsed.as_secs_f64(),
    );

    // ── Read back completed ranges ────────────────────────────────────────────

    // Sort by disk_offset so we read the file in order.
    all_ranges.sort_by_key(|r| r.disk_offset);

    debug!("\n--- {} completed flush range(s) ---", all_ranges.len());
    println!("\n--- {} completed flush range(s) ---", all_ranges.len());

    let mut read_file = File::open("data_output.log")?;

    for (i, range) in all_ranges.iter().enumerate() {
        let mut buf = vec![0u8; range.sealed_len];

        read_file.seek(SeekFrom::Start(range.disk_offset))?;
        read_file.read_exact(&mut buf)?;

        let text = String::from_utf8_lossy(&buf);

        debug!("[range {i}] disk_offset={} sealed_len={}", range.disk_offset, range.sealed_len);
        println!("[range {i}] disk_offset={} sealed_len={}", range.disk_offset, range.sealed_len);
        debug!("{}", text);
    }

    // ── Also log every line through flexi_logger ──────────────────────────────

    println!("\nReading data file and logging to flexi_logger...");

    let log_file = File::open("data_output.log")?;
    let reader   = BufReader::new(log_file);
    let mut line_count = 0;

    for line in reader.lines() {
        if let Ok(content) = line {
            debug!("{}", content);
            line_count += 1;
        }
    }

    println!("Logged {} lines to flexi_logger", line_count);
    drop(_logger);
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("\nCheck logs/verification.log for all messages!");

    Ok(())
}