use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};

use crate::{
    BufferError, BufferMsg, FLUSH_IN_PROGRESS_BIT, FOUR_KB_BLOCK, FlushBuffer, OFFSET_SHIFT,
    SEALED_BIT, quik_io::QuikIO, state_sealed, state_writers,
};

/// A fixed-size ring of [`FlushBuffer`]s that amortises writes into batched
/// sequential I/O.
///
/// The ring maintains a single *current* buffer pointer that all threads write
/// into concurrently.  When the current buffer is full it is sealed, a fresh
/// buffer is selected from the ring, and the sealed buffer is optionally dispatched to
/// the configured [`QuikIO`] for an `io_uring` write.
///
/// New LSS address slots are assigned at seal time via a single atomic fetch-add on
/// [`next_address_range`](Self), ensuring that no two buffers ever map to the
/// same region of the backing file even when flushes complete out of order.
///
/// # Ring Exhaustion
///
/// If all buffers in the ring are sealed or being flushed when a rotation is
/// needed, [`rotate_after_seal`](Self::rotate_after_seal) returns
/// [`BufferError::RingExhausted`].  Callers should back off and poll the
/// completion to free up buffers.
pub struct BufferRing {
    /// Pointer to the buffer currently accepting writes.
    ///
    /// Updated atomically via CAS during rotation.  The pointed-to buffer is
    /// guaranteed to be valid for the lifetime of the ring because all buffers
    /// are owned by `ring` and the ring is `Pin`ned.
    current_buffer: AtomicPtr<FlushBuffer>, // TODO getters and Setters

    /// Pinned, heap-allocated array of all buffers.
    ///
    /// `Pin` ensures the buffers never move in memory, which is required
    /// because `current_buffer` holds raw pointers into this slice, and the
    /// `io_uring` SQEs hold raw pointers into the backing allocations.
    ring: Pin<Box<[Arc<FlushBuffer>]>>,

    /// Index of the next candidate buffer during rotation.
    next_index: AtomicUsize,

    /// Monotonically increasing LSS slot counter.
    ///
    /// Incremented by fetch-add at seal time; the resulting value is stored as
    /// the sealed buffer's `local_address`.
    next_address_range: AtomicUsize, // TODO getters and Setters dont make pub

    /// Optional flush dispatcher.  `None` in test mode — buffers are reset
    /// immediately without dispatching any `io_uring` writes.
    store: Option<Arc<QuikIO>>,

    /// Whether to automatically flush buffers when they are sealed.
    /// If false, users must manually flush the buffer's contents
    auto_flush: bool,

    auto_rotate: bool,

    size: usize,
}

pub struct BufferRingOptions {
    capacity: usize,
    buffer_size: usize,
    io_instance: Option<Arc<QuikIO>>,
    auto_flush: bool,
    auto_rotate: bool,
}

/// Configuration options for a `BufferRing` instance.
///
/// `BufferRingOptions` provides a builder-style API to configure the behavior and properties
/// of a buffer ring. All configuration methods consume and return `self`, allowing for
/// convenient method chaining.
///
/// # Examples
///
/// ```ignore
/// let options = BufferRingOptions::new(100, 1024)
///     .auto_flush(true)
///     .auto_rotate(true);
/// ```
impl BufferRingOptions {
    /// Creates a new `BufferRingOptions` with the specified capacity and buffer size.
    ///
    /// # Returns
    ///
    /// A new `BufferRingOptions` instance with default settings
    pub fn new() -> Self {
        Self {
            capacity: 0,
            buffer_size: 0,
            io_instance: None,
            auto_flush: true,
            auto_rotate: true,
        }
    }

    /// Sets the capacity for this buffer ring.
    ///
    pub fn capacity(&mut self, cap: usize) -> &mut Self {
        self.capacity = cap;
        self
    }

    /// Sets the size of for this buffer ring.
    ///
    pub fn buffer_size(&mut self, buffer_size: usize) -> &mut Self {
        let size = buffer_size.next_multiple_of(buffer_size);
        self.capacity = size;
        self
    }

    /// Sets the I/O instance for this buffer ring.
    ///
    /// # Arguments
    ///
    /// * `io` - An `Arc`-wrapped `QuikIO` instance to use for I/O operations
    ///
    pub fn io_instance(&mut self, io: Arc<QuikIO>) -> &mut Self {
        self.io_instance = Some(io);
        self
    }

    /// Enables or disables automatic flushing.
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to enable auto-flush, `false` to disable
    ///

    pub fn auto_flush(&mut self, enabled: bool) -> &mut Self {
        self.auto_flush = enabled;
        self
    }

    /// Enables or disables automatic buffer rotation.
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to enable auto-rotate, `false` to disable
    ///
    pub fn auto_rotate(&mut self, enabled: bool) -> &mut Self {
        self.auto_rotate = enabled;
        self
    }
}

impl BufferRing {
    /// Create a ring of `num_of_buffer` buffers, each `buffer_size` bytes,
    /// with **no** flush dispatcher attached.
    ///
    /// Intended for unit tests that exercise the ring's concurrency primitives
    /// without requiring a real `io_uring` instance or backing file.  In this
    /// mode, sealed buffers are reset immediately after flush is triggered,
    /// keeping the ring from stalling.
    pub fn with_options(options: BufferRingOptions) -> BufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..options.capacity)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, options.buffer_size)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        BufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            size: options.capacity,
            next_address_range: AtomicUsize::new(0),
            store: options.io_instance,
            auto_flush: options.auto_flush,
            auto_rotate: options.auto_rotate,
        }
    }

    /// Write `payload` into `current` at the byte offset described by
    /// `reserve_result`.
    ///
    /// Handles all outcomes of a prior [`reserve_space`](FlushBuffer::reserve_space)
    /// call:
    ///
    /// * **`Ok(offset)`** — copy the payload, decrement the writer count, and
    ///   trigger a flush if this thread is the last writer in a sealed buffer.
    /// * **`Err(InsufficientSpace)`** — seal the buffer, rotate the ring, and
    ///   initiate a flush if this thread wins the flush-in-progress CAS race.
    /// * **`Err(EncounteredSealedBuffer)`** — propagated to the caller; the
    ///   ring has already rotated and the caller should retry on the new buffer.
    ///
    /// # Errors
    ///
    /// Propagates [`BufferError`] variants from the ring.
    pub fn put(
        &self,
        current: &FlushBuffer,
        reserve_result: Result<usize, BufferError>,
        payload: &[u8],
    ) -> Result<BufferMsg, BufferError> {
        match reserve_result {
            Err(BufferError::InsufficientSpace) => {
                // Seal the buffer. Only the first thread to set the bit "owns" the seal.
                let prev = current.state.fetch_or(SEALED_BIT, Ordering::AcqRel);

                if prev & SEALED_BIT != 0 {
                    // Someone else already sealed it → just retry on new current buffer
                    return Err(BufferError::EncounteredSealedBuffer);
                }

                // We are the sealer. Claim unique LSS slot.
                let padded = current.size().next_multiple_of(FOUR_KB_BLOCK);
                let slot = self.incrment_address(padded, Ordering::Acquire);
                current.local_address.store(slot, Ordering::Release);

                if self.auto_rotate {
                    let _ = self.rotate_after_seal(current.pos); // ignore error for now
                }

                // If there are no writers left *right now*, we become the flusher.
                let state_now = current.state.load(Ordering::Acquire);
                if state_writers(state_now) == 0 {
                    let before = current.set_flush_in_progress();
                    if before & FLUSH_IN_PROGRESS_BIT == 0 {
                        match self.store.as_ref() {
                            Some(_) if self.auto_flush => {
                                let flush_buffer = self.ring.get(current.pos).unwrap().clone();
                                self.flush(&flush_buffer);
                            }
                            _ => self.reset_buffer(current),
                        }
                        return Ok(BufferMsg::SuccessfullWriteFlush);
                    }
                }

                // Still writers active → they will flush when they decrement.
                // Caller must retry on the new current buffer.
                Err(BufferError::EncounteredSealedBuffer)
            }
            Err(BufferError::EncounteredSealedBuffer) => {
                return Err(BufferError::EncounteredSealedBuffer);
            }

            Err(e) => return Err(e),

            Ok(offset) => {
                current.write(offset, payload);

                let prev = current.decrement_writers();

                // Note: Atomic operations always yeild previous values
                let was_last_writer = state_writers(prev) == 1;
                let was_sealed = state_sealed(prev);

                if was_last_writer && was_sealed {
                    let prev = current.set_flush_in_progress();
                    if prev & FLUSH_IN_PROGRESS_BIT == 0 {
                        match self.store.as_ref() {
                            Some(_) if self.auto_flush => {
                                let flush_buffer = self.ring.get(current.pos).unwrap().clone();
                                self.flush(&flush_buffer);
                            }
                            _ => self.reset_buffer(current),
                        }
                        return Ok(BufferMsg::SuccessfullWriteFlush);
                    }
                }

                return Ok(BufferMsg::SuccessfullWrite);
            }
        }
    }

    /// Explicitly dispatch a buffer to stable storage asynchronously.
    ///
    /// Sets the flush-in-progress bit and submits the buffer to the configured
    /// [`QuickIO`]. In test mode (no dispatcher configured), the buffer
    /// is reset immediately so the ring does not stall waiting for a CQE that
    /// will never arrive.
    ///
    /// This method is called internally by [`put`](Self::put) when `auto_flush` is enabled,
    /// but can also be called manually for custom buffer protocols when `auto_flush` is disabled.
    ///
    /// # Important Notes for Manual Flushing
    ///
    /// When `auto_flush` is false, you assume responsibility for:
    ///
    /// 1. **Deadlock Prevention**: The ring will exhaust if all buffers are sealed and none
    ///    are flushed. Ensure you flush regularly to prevent [`BufferError::RingExhausted`]
    ///    errors.
    ///
    /// 2. **Ordering Guarantees**: Flush operations are asynchronous and submitted to an
    ///    external dispatcher. If you need guarantees about write ordering to stable storage,
    ///    you must coordinate with your [`QuickIO`] implementation.
    ///
    /// 3. **Buffer Lifecycle**: A buffer remains locked in flush-in-progress state until
    ///    [`reset_buffer`](Self::reset_buffer) is called. The dispatcher is responsible
    ///    for calling reset after I/O completion. Failing to reset will cause ring exhaustion.
    ///
    /// 4. **Serialization Responsibility**: This ring manages storage buffering only. You must
    ///    ensure all data is properly serialized into buffers before initiating flushes.
    ///
    /// 5. **Thread Safety**: Concurrent flushes from multiple threads are safe, but concurrent
    ///    writes to the same buffer after it is sealed may cause data corruption. The ring
    ///    rotates to prevent this, but manual protocols must enforce their own constraints.
    ///
    /// ```
    pub fn flush(&self, buffer: &FlushBuffer) {
        buffer.set_flush_in_progress();

        match self.store.as_ref() {
            Some(store) => {
                store.submit_buffer(buffer);
            }
            None => {
                self.reset_buffer(buffer);
            }
        }
    }

    /// Blocks and waits untill all previously flush data has been persisten
    pub fn f_sync(&self, buffer: &FlushBuffer) {
        buffer.set_flush_in_progress();

        match self.store.as_ref() {
            Some(store) => {
                store.submit_buffer(buffer);
                store.sync_data().expect("Drained Submission Queue");
                self.reset_buffer(buffer);
            }
            None => {
                self.reset_buffer(buffer);
            }
        }
    }

    /// Rotate the ring's current buffer pointer away from the buffer at
    /// `sealed_pos`.
    ///
    /// Scans the ring for the next available (unsealed, not flushing) buffer and
    /// swaps `current_buffer` to point at it via CAS.  If no available buffer is
    /// found, returns [`BufferError::RingExhausted`].
    ///
    /// If `current_buffer` has already been rotated by another thread (i.e. it
    /// no longer points at `sealed_pos`), returns `Ok(())` immediately.
    pub fn rotate_after_seal(&self, sealed_pos: usize) -> Result<(), BufferError> {
        let current = self.current_buffer.load(Ordering::Acquire);
        let current_ref = unsafe { current.as_ref().ok_or(BufferError::InvalidState)? };

        if current_ref.pos != sealed_pos {
            return Ok(());
        }

        let ring_len = self.ring.len();

        for _ in 0..ring_len {
            let raw = self.next_index.fetch_add(1, Ordering::AcqRel);
            let next_index = raw % ring_len;
            let new_buffer = &self.ring[next_index];

            if new_buffer.is_available() {
                let _ = self.current_buffer.compare_exchange(
                    current,
                    Arc::as_ptr(new_buffer) as *const FlushBuffer as *mut FlushBuffer,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Ok(());
            }
        }

        Err(BufferError::RingExhausted)
    }

    /// Reset a buffer's state after it has been flushed to storage.
    ///
    /// This clears the SEALED_BIT, FLUSH_IN_PROGRESS_BIT, and write offset, making the
    /// buffer available for reuse in the ring. Called automatically when `auto_flush` is enabled
    /// (after I/O completion via the [`QuickIO`] dispatcher).
    ///
    /// When implementing manual flush protocols (`auto_flush` disabled), you typically do NOT
    /// call this directly. Instead, your I/O completion handler (from the [`QuickIO`])
    /// should call this to re-enable the buffer.
    ///
    /// # Critical Notes
    ///
    /// - **Responsibility**: When `auto_flush` is false, your external I/O completion handler
    ///   must call this method after the buffer's data has been successfully persisted.
    /// - **Timing**: Calling this too early (before I/O completion) risks data loss.
    /// - **Atomicity**: Uses CAS loops internally and will retry until successful.
    ///
    /// # Example: I/O Completion Handler
    ///
    /// ```ignore
    /// // In your QuickIO dispatcher or io_uring completion handler:
    /// fn on_io_completion(buffer: &FlushBuffer) {
    ///     // I/O is complete, data is now on disk
    ///     ring.reset_buffer(buffer);  // Re-enable for the ring
    /// }
    /// ```
    pub fn reset_buffer(&self, buffer: &FlushBuffer) {
        loop {
            let flushed_buffer_state = buffer.state.load(Ordering::Acquire);

            const OFFSET_MASK: usize = usize::MAX << OFFSET_SHIFT;
            let reset = flushed_buffer_state & !(SEALED_BIT | FLUSH_IN_PROGRESS_BIT | OFFSET_MASK);

            if buffer
                .state
                .compare_exchange(
                    flushed_buffer_state,
                    reset,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    /// Internal completion queue processing.
    ///
    /// Drains all available CQEs and re-submits any failed writes
    pub fn check_cque(&self) -> Result<(), String> {
        if let Some(store) = &self.store {
            let cqes = store.cqe();

            if cqes.is_empty() {
                return Ok(());
            }
            eprintln!(
                "[check_cque] cqe results: {:?}",
                cqes.iter().map(|c| c.result()).collect::<Vec<_>>()
            );

            for cqe in cqes {
                let ptr = cqe.user_data() as *const FlushBuffer;
                let buffer: &FlushBuffer = unsafe { &*ptr };

                if cqe.result() < 0 {
                    // Retry failed write
                    let sqe = unsafe {
                        (*buffer.sqe.get())
                            .as_ref()
                            .expect("stored SQE must be present on retry")
                    };
                    let mut ring = store.ring();
                    unsafe {
                        let _ = ring.submission().push(&sqe);
                    }
                    let _ = ring.submit();
                } else {
                    // Write completed successfully — release the buffer back to the ring
                    self.reset_buffer(buffer);
                }
            }

            return Ok(());
        }

        return Err("Store Not present".to_string());
    }


    /// Atomically loads the address range
    pub fn next_address(&self, ordering: Ordering) -> usize {
        self.next_address_range.load(ordering)
    }

    /// Atomically increments the lss address range of a flush buffer.
    pub fn incrment_address(&self, val: usize, ordering: Ordering) -> usize {
        self.next_address_range.fetch_add(val, ordering)
    }

    /// Get a reference to the current active buffer.
    ///
    /// # Safety
    ///
    /// The returned reference is valid only for the current snapshot. The ring may rotate
    /// to a different buffer at any time if the current one is sealed. Use this method only
    /// when you need to inspect buffer state for custom protocols.
    pub fn current_buffer(&self, ordering: Ordering) -> &'static FlushBuffer {
        let ptr = self.current_buffer.load(ordering);
        unsafe { ptr.as_ref().unwrap() }
    }


    /// Gets the ring size
    pub fn ring_size(&self) -> usize {
        self.size
    }
}
#[cfg(test)]
mod tests {

    use super::*;

    use std::{
        fs::OpenOptions,
        os::unix::fs::OpenOptionsExt,
        sync::{Arc, Barrier, Mutex, atomic::AtomicBool},
        thread,
        time::{Duration, Instant},
    };

    /// Very small, very lightweight, very unimpressive Linear Congruential Generator for deterministic
    /// pseudorandom number generation in tests.
    /// source: https://en.wikipedia.org/wiki/Linear_congruential_generator
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_usize(&mut self, bound: usize) -> usize {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 33) as usize) % bound
        }
    }

    const TEST_RING_SIZE: usize = 4;
    const OPS_PER_THREAD: usize = 2_000;

    /// Payload sizes ranging from tiny to near-capacity.
    const SIZES: &[usize] = &[
        1, 2, 4, 7, 8, 15, 16, 32, 64, 100, 128, 200, 256, 512, 1024, 2048, 4090, 4095, 4096,
    ];

    /// Build a recognisable, size-stamped payload.
    fn make_payload(tag: &str, size: usize) -> Vec<u8> {
        let meta = format!("[{tag}:{size}]");
        let mut buf = vec![0xAA_u8; size];
        let n = meta.len().min(size);
        buf[..n].copy_from_slice(&meta.as_bytes()[..n]);
        buf
    }

    // =========================================================================
    // Retry helper
    //
    // The ring does not retry internally — that is the caller's responsibility
    // (mapping table in production, this helper in tests).
    //
    // Loop:
    //   1. Load current buffer.
    //   2. Call reserve_space.
    //   3. Pass result into put.
    //   4. Retry on transient errors (FailedReservation, EncounteredSealedBuffer,
    //      ActiveUsers).
    //   5. Any other outcome is final.
    // =========================================================================
    fn put_with_retry(ring: &BufferRing, payload: &[u8]) -> Result<BufferMsg, BufferError> {
        loop {
            let _ = ring.check_cque();
            let current = unsafe {
                ring.current_buffer
                    .load(Ordering::Acquire)
                    .as_ref()
                    .ok_or(BufferError::InvalidState)?
            };

            let reserve_result = current.reserve_space(payload.len());

            match &reserve_result {
                Err(BufferError::FailedReservation) => continue,
                Err(BufferError::EncounteredSealedBuffer) => continue,
                _ => {}
            }

            match ring.put(current, reserve_result, payload) {
                Err(BufferError::ActiveUsers) => continue,
                Err(BufferError::EncounteredSealedBuffer) => {
                    let _ = ring.check_cque();
                    std::thread::yield_now();
                    continue;
                }
                Err(BufferError::RingExhausted) => {
                    let _ = ring.check_cque();
                    std::thread::yield_now();
                    continue;
                }
                other => return other,
            }
        }
    }

    use tempfile::NamedTempFile;

    use crate::{ONE_MEGABYTE_BLOCK, quik_io::QuikIO};

    const NUM_THREADS_SMALL: usize = 2;
    const NUM_THREADS_MEDIUM: usize = 4;
    const NUM_THREADS_LARGE: usize = 8;

    #[test]
    fn writer_test_small() {
        multi_threaded_stress_writer(NUM_THREADS_SMALL);
    }

    #[test]
    fn writer_test_medium() {
        multi_threaded_stress_writer(NUM_THREADS_MEDIUM);
    }

    #[test]
    fn writer_test_large() {
        multi_threaded_stress_writer(NUM_THREADS_LARGE);
    }

    fn multi_threaded_stress_writer(num_threads: usize) {
        let temp_file = NamedTempFile::new().unwrap();

        let file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                // O_DIRECT bypasses the kernel page cache.
                // INVARIANT: every buffer passed to read/write must be aligned to
                // FOUR_KB_PAGE — upheld by Buffer::new_aligned in flush_buffer.rs.
                .custom_flags(libc::O_DIRECT)
                .open(temp_file.path())
                .unwrap(),
        );

        let quickio = QuikIO::new(file);
        let ring = Arc::new(BufferRing::with_options(BufferRingOptions {
            capacity: TEST_RING_SIZE,
            buffer_size: ONE_MEGABYTE_BLOCK,
            io_instance: Some(quickio.into()),
            auto_flush: true,
            auto_rotate: true,
        }));

        let watchdog_ring = Arc::clone(&ring);
        let watchdog_done = Arc::new(AtomicBool::new(false));
        let watchdog_done_clone = Arc::clone(&watchdog_done);

        let watchdog = thread::spawn(move || {
            let mut tick = 0u64;
            while !watchdog_done_clone.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(500));
                tick += 1;

                let current_ptr = watchdog_ring.current_buffer.load(Ordering::Acquire);
                let current = unsafe { current_ptr.as_ref().unwrap() };
                let state = current.state();

                eprintln!(
                    "[watchdog tick={tick}] current_buffer pos={pos} | \
                 offset={offset} writers={writers} sealed={sealed} flushing={flushing}",
                    pos = current.pos,
                    offset = state.offset(),
                    writers = state.n_writers(),
                    sealed = state.sealed(),
                    flushing = state.flushing(),
                );

                // Dump every buffer in the ring
                for (i, buf) in watchdog_ring.ring.iter().enumerate() {
                    let s = buf.state();
                    eprintln!(
                        "  [buf {i}] offset={} writers={} sealed={} flushing={} addr={}",
                        s.offset(),
                        s.n_writers(),
                        s.sealed(),
                        s.flushing(),
                        buf.local_address(Ordering::Acquire),
                    );
                }
            }
            eprintln!("[watchdog] shutting down after {tick} ticks");
        });
        // ─────────────────────────────────────────────────────────────────────────

        let barrier = Arc::new(Barrier::new(num_threads));
        let total_writes = Arc::new(AtomicUsize::new(0));
        let total_flushes = Arc::new(AtomicUsize::new(0));
        let start_times = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<thread::JoinHandle<()>> = (0..num_threads)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);
                let total_writes = Arc::clone(&total_writes);
                let total_flushes = Arc::clone(&total_flushes);
                let start_times = Arc::clone(&start_times);

                let seed = 0x1234_5678_u64
                    .wrapping_add(tid as u64)
                    .wrapping_mul(0xDEAD_CAFE);

                thread::spawn(move || {
                    let mut rng = Lcg::new(seed);
                    let mut local_writes = 0usize;
                    let mut local_flushes = 0usize;

                    barrier.wait();
                    start_times.lock().unwrap().push(Instant::now());

                    for op in 0..OPS_PER_THREAD {
                        let size = SIZES[rng.next_usize(SIZES.len())];
                        let payload = make_payload(&format!("T{tid}:O{op:04}"), size);

                        match put_with_retry(&ring, &payload) {
                            Ok(BufferMsg::SuccessfullWrite) => local_writes += 1,
                            Ok(BufferMsg::SuccessfullWriteFlush) => {
                                local_writes += 1;
                                local_flushes += 1;
                            }
                            other => panic!("thread {tid} op {op}: unexpected {other:?}"),
                        }
                    }

                    total_writes.fetch_add(local_writes, Ordering::Relaxed);
                    total_flushes.fetch_add(local_flushes, Ordering::Relaxed);
                })
            })
            .collect();

        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .unwrap_or_else(|_| panic!("worker thread {tid} panicked"));
        }

        // Signal watchdog to stop and wait for it
        watchdog_done.store(true, Ordering::Relaxed);
        watchdog.join().unwrap();

        let join_time = Instant::now();
        let writes = total_writes.load(Ordering::Relaxed);
        let flushes = total_flushes.load(Ordering::Relaxed);
        let earliest_start = start_times.lock().unwrap().iter().copied().min().unwrap();
        let elapsed = join_time.duration_since(earliest_start);

        println!(
            "multi_threaded_stress({num_threads} threads): {writes} writes, {flushes} flushes \
         in {elapsed:.2?} ({:.0} ops/s)",
            writes as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(
            writes,
            num_threads * OPS_PER_THREAD,
            "total writes should equal num_threads * OPS_PER_THREAD"
        );
    }

    #[test]
    fn multi_threaded_test_small() {
        multi_threaded_stress_helper(NUM_THREADS_SMALL);
    }

    #[test]
    fn multi_threaded_test_medium() {
        multi_threaded_stress_helper(NUM_THREADS_MEDIUM);
    }

    #[test]
    fn multi_threaded_test_large() {
        multi_threaded_stress_helper(NUM_THREADS_LARGE);
    }
    fn multi_threaded_stress_helper(num_threads: usize) {
        let ring = Arc::new(BufferRing::with_options(BufferRingOptions {
            capacity: TEST_RING_SIZE,
            buffer_size: ONE_MEGABYTE_BLOCK,
            io_instance: None,
            auto_flush: false,
            auto_rotate: true,
        }));

        let watchdog_ring = Arc::clone(&ring);
        let watchdog_done = Arc::new(AtomicBool::new(false));
        let watchdog_done_clone = Arc::clone(&watchdog_done);

        let watchdog = thread::spawn(move || {
            let mut tick = 0u64;
            while !watchdog_done_clone.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(500));
                tick += 1;

                let current_ptr = watchdog_ring.current_buffer.load(Ordering::Acquire);
                let current = unsafe { current_ptr.as_ref().unwrap() };
                let state = current.state();

                eprintln!(
                    "[watchdog tick={tick}] current_buffer pos={pos} | \
                 offset={offset} writers={writers} sealed={sealed} flushing={flushing}",
                    pos = current.pos,
                    offset = state.offset(),
                    writers = state.n_writers(),
                    sealed = state.sealed(),
                    flushing = state.flushing(),
                );

                // Dump every buffer in the ring
                for (i, buf) in watchdog_ring.ring.iter().enumerate() {
                    let s = buf.state();
                    eprintln!(
                        "  [buf {i}] offset={} writers={} sealed={} flushing={} addr={}",
                        s.offset(),
                        s.n_writers(),
                        s.sealed(),
                        s.flushing(),
                        buf.local_address(Ordering::Acquire),
                    );
                }
            }
            eprintln!("[watchdog] shutting down after {tick} ticks");
        });
        // ─────────────────────────────────────────────────────────────────────────

        let barrier = Arc::new(Barrier::new(num_threads));
        let total_writes = Arc::new(AtomicUsize::new(0));
        let total_flushes = Arc::new(AtomicUsize::new(0));
        let start_times = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<thread::JoinHandle<()>> = (0..num_threads)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);
                let total_writes = Arc::clone(&total_writes);
                let total_flushes = Arc::clone(&total_flushes);
                let start_times = Arc::clone(&start_times);

                let seed = 0x1234_5678_u64
                    .wrapping_add(tid as u64)
                    .wrapping_mul(0xDEAD_CAFE);

                thread::spawn(move || {
                    let mut rng = Lcg::new(seed);
                    let mut local_writes = 0usize;
                    let mut local_flushes = 0usize;

                    barrier.wait();
                    start_times.lock().unwrap().push(Instant::now());

                    for op in 0..OPS_PER_THREAD {
                        let size = SIZES[rng.next_usize(SIZES.len())];
                        let payload = make_payload(&format!("T{tid}:O{op:04}"), size);

                        match put_with_retry(&ring, &payload) {
                            Ok(BufferMsg::SuccessfullWrite) => local_writes += 1,
                            Ok(BufferMsg::SuccessfullWriteFlush) => {
                                local_writes += 1;
                                local_flushes += 1;
                            }
                            other => panic!("thread {tid} op {op}: unexpected {other:?}"),
                        }
                    }

                    total_writes.fetch_add(local_writes, Ordering::Relaxed);
                    total_flushes.fetch_add(local_flushes, Ordering::Relaxed);
                })
            })
            .collect();

        for (tid, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .unwrap_or_else(|_| panic!("worker thread {tid} panicked"));
        }

        // Signal watchdog to stop and wait for it
        watchdog_done.store(true, Ordering::Relaxed);
        watchdog.join().unwrap();

        let join_time = Instant::now();
        let writes = total_writes.load(Ordering::Relaxed);
        let flushes = total_flushes.load(Ordering::Relaxed);
        let earliest_start = start_times.lock().unwrap().iter().copied().min().unwrap();
        let elapsed = join_time.duration_since(earliest_start);

        println!(
            "multi_threaded_stress({num_threads} threads): {writes} writes, {flushes} flushes \
         in {elapsed:.2?} ({:.0} ops/s)",
            writes as f64 / elapsed.as_secs_f64()
        );

        assert_eq!(
            writes,
            num_threads * OPS_PER_THREAD,
            "total writes should equal num_threads * OPS_PER_THREAD"
        );
    }
}
