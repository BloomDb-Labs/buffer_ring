//! # QuickIO — `io_uring`-backed Write Dispatchers
//!
//! This module defines the two write strategies LLAMA uses to flush sealed
//! [`crate::FlushBuffer`]s to the log-structured backing file:
//!
//! | Strategy                       | Type                                 | Ordering |
//! |--------------------------------|--------------------------------------|----------|
//! | Tail-Localised Writes          | [`QuickIO::TailLocalized`]           | Parallel |
//! | Strictly Serialised Writes     | [`QuickIO::Searalized`]              | `IO_LINK`|
//!
//! Both strategies are backed by the same [`BackingStore`] struct; the difference
//! lies in the `io_uring` submission-queue flags applied at dispatch time.
//!
//! ## Why Two Strategies?
//!
//! ### Tail-Localised Writes
//!
//! Append-only write patterns deliver substantial throughput improvements on
//! both spinning-disk and SSD storage. LLAMA exploits this by
//! staging writes in a ring of [`ONE_MEGABYTE_BLOCK`] Buffers.  Each buffer
//! is assigned a unique, non-overlapping slot in the LSS address space at seal
//! time; once sealed, buffers are flushed independently with no synchronisation
//! between them.
//!
//! Because slots are claimed atomically (fetch-add) but flushed concurrently, a
//! buffer sealed *later* may land on disk *before* an earlier one.  This means
//! flushes are **tail-localised** rather than strictly sequential. Assuming that all 
//! writes are completied within a single rotation , the maximum write distance from the 
//! logical tail is bounded by:
//!
//! ```text
//! max_distance = RING_SIZE × ONE_MEGABYTE_BLOCK
//! ```
//!
//! ### Serialised Writes
//!
//! For workloads that require strict append order (e.g. WAL segments, recovery
//! logs), [`WriteMode::SerializedWrites`] applies `IO_LINK` to the SQE chain.
//! The kernel will not begin the *n+1*th write until the *n*th has completed,
//! enforcing submission-order on disk at the cost of reduced parallelism.
//!
//! ## Completion Handling
//!
//! LLAMA deliberately avoids a dedicated watchdog thread.  Instead, a calling
//! thread inspects the completion queue at a well-defined point on the write
//! path via [`BackingStore::cqueue`]. 
//! 
//! ## `O_DIRECT` Alignment Invariant
//!
//! All buffers submitted through this module **must** be aligned to
//! [`ONE_MEGABYTE_BLOCK`] and their lengths must be a multiple of the device's
//! logical block size.  This invariant is upheld by `Buffer::new_aligned`
//! inside [`crate::BufferRing`].

use io_uring::{opcode, squeue, types, IoUring};

use std::{
    fs::File,
    io,
    os::fd::AsRawFd,
    sync::Arc,
    cell::UnsafeCell,
};


#[allow(unused_imports)]
use crate::ONE_MEGABYTE_BLOCK;

/// Type alias for the submit queue entry storage used by flush operations.
pub type SubmitQueueEntry = UnsafeCell<Option<squeue::Entry>>;
/// Trait for buffers that can be submitted for flushing.
///
/// Implementors must provide the data to write, the offset, user data, and
/// a place to store the SQE for potential re-submission.
pub trait FlushableBuffer {
    /// Get the data to write.
    fn buffer_data(&self) -> &[u8];
    /// Get the byte offset in the file.
    fn offset(&self) -> u64;
    /// Get the user data for the SQE.
    fn user_data(&self) -> u64;
    /// Get the submit queue entry storage.
    fn submit_entry(&self) -> &SubmitQueueEntry;
}
/// Flush Buffers must adherer to Strict Serialized Ordered Writes
#[allow(unused)]
const SERIALIZED_ORDERING: u8 = 0;

/// Flag Buffers are permitted to write within a localized region
/// within [`RING_SIZE`] × [`FOUR_KB_PAGE`] of the tail
#[allow(unused)]
const LOCALIZED_WRITES: u8 = 1;

/// A shared, mutex-protected `io_uring` handle.
///
/// The `Mutex` is from [`parking_lot`] and is fair, making it suitable for use
/// across many short-lived critical sections on the write path.
pub type SharedAsyncFileWriter = Arc<parking_lot::Mutex<IoUring>>;

/// Controls the `io_uring` submission-queue flags used when dispatching writes.
///
/// Choose [`TailLocalizedWrites`](WriteMode::TailLocalizedWrites) for maximum
/// throughput and choose [`SerializedWrites`](WriteMode::SerializedWrites) when
/// strict append ordering is required (e.g. WAL segments).
///
/// # Examples
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use flush_buffer_ring::flush_behaviour::WriteMode;
///
/// // High-throughput ingestion path — writes may land out of order within
/// // RING_SIZE × FOUR_KB_PAGE of the tail.
/// let mode = WriteMode::TailLocalizedWrites;
///
/// // Recovery-log path — each write completes before the next begins.
/// let mode = WriteMode::SerializedWrites;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum WriteMode {
    /// Parallel localized writes
    TailLocalizedWrites,
    /// Serialized ordered writes — drain ordering enforced
    SerializedWrites,
}

/// Unified `io_uring` backing store — handles both localised and serialised flush strategies.
///
/// [`BackingStore`] wraps a shared [`IoUring`] instance and an `O_DIRECT` [`File`]
/// handle.  The concrete write behaviour (parallel vs. ordered) is determined by
/// the [`WriteMode`] stored at construction time.
///
/// In normal operation callers should not use `BackingStore` directly; instead,
/// interact with the store through [`QuickIO`].
pub struct BackingStore {
    /// Shared `O_DIRECT` file handle — the LSS backing file.
    store: Arc<File>,
    /// Shared `io_uring` instance.  Protected by a [`parking_lot::Mutex`] so
    /// that multiple threads can submit SQEs.
    flusher: SharedAsyncFileWriter,
    /// Determines SQE flags applied to every write submission.
    mode: WriteMode,
}

impl BackingStore {
    /// Create a new `BackingStore` from an existing `io_uring` instance and file handle.
    ///
    /// # Arguments
    ///
    /// * `io_uring`    — Shared, mutex-protected `io_uring` ring.
    /// * `file_handle` — `O_DIRECT` file handle to the LSS backing file.
    /// * `mode`        — Write ordering mode.
    pub fn new(io_uring: SharedAsyncFileWriter, file_handle: Arc<File>, mode: WriteMode) -> Self {

        
        Self {
            flusher: io_uring,
            store: file_handle,
            mode,
        }
    }

    /// Submit a **fire-and-forget** write for `buffer_data` at byte offset `at`.
    ///
    /// Returns immediately after the SQE is pushed to the submission ring; the
    /// kernel picks it up asynchronously.  Poll completions via
    /// [`BackingStore::cqueue`].
    ///
    /// The SQE is also stored inside `submit_entry` so that a
    /// failed completion can re-submit the exact same write without
    /// re-constructing it.
    ///
    /// # Arguments
    ///
    /// * `buffer_data` — Aligned slice covering exactly the bytes to write
    ///                   (`0..used_bytes`).
    /// * `at`          — Byte offset in the backing file (`slot × FOUR_KB_PAGE`).
    /// * `buffer_ptr`   — Raw pointer to the buffer cast to `u64`, stored as the
    ///                   SQE's `buffer_ptr` so the completion handler can recover
    ///                   the buffer without an extra lookup.
    /// * `submit_entry` — Storage for the submitted SQE for potential re-submission.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the submission ring is full or if
    /// `io_uring::submit()` fails.
    ///
    /// # Safety
    ///
    /// The pointed-to memory in `buffer_data` must remain valid and unmodified until the
    /// corresponding CQE is observed.
    pub fn submit(
        &self,
        buffer_data: &[u8],
        at: u64,
        buffer_ptr: u64,
        submit_entry: &SubmitQueueEntry,
    ) -> io::Result<()> {
        let flags = match self.mode {
            // Parallel writes — kernel may reorder freely.
            // Safe because each buffer owns a non-overlapping LSS address range.
            WriteMode::TailLocalizedWrites => squeue::Flags::empty(),

            // Serialized writes — each write is linked to the next.
            // Kernel will not start the next write until this one completes.
            // Ordering is enforced by submission order, not a drain barrier.
            WriteMode::SerializedWrites => squeue::Flags::IO_LINK,
        };

        let sqe = opcode::Write::new(
            types::Fd(self.store.as_raw_fd()),
            buffer_data.as_ptr(),
            buffer_data.len() as u32,
        )
        // Slots are flawed they assume buffers will be filled to capacity
        .offset(at)
        .build()
        .flags(flags)
        .user_data(buffer_ptr);

        let mut ring = self.flusher.lock();

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| io::Error::other("SQ full"))?;

            *submit_entry.get() = Some(sqe);
        }

        // submit() returns immediately — kernel picks it up asynchronously.
        // We do NOT call submit_and_wait() here.
        ring.submit()?;

        Ok(())
    }

    /// Acquire a mutex guard giving exclusive access to the underlying `io_uring`
    /// instance, including its completion queue.
    ///
    /// The guard is held for the duration of the caller's critical section; keep
    /// it as short-lived as possible to avoid starving the write path.

    pub fn cqueue(&self) -> parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, IoUring> {
        let flusher_ring = self.flusher.lock();
        flusher_ring
    }
}

/// The top-level flush dispatcher — selects between parallel and serialised write modes.
///
/// `QuickIO` is an enum over the two variants of [`BackingStore`] so that the
/// store can branch once at construction time and then call the same interface
/// everywhere on the hot path.
///
/// # Variants
///
/// * [`Searalized`](QuickIO::Searalized) — wraps a [`BackingStore`] in
///   [`WriteMode::SerializedWrites`].  Use for WAL segments or any workload that
///   requires each write to complete before the next begins.
/// * [`TailLocalized`](QuickIO::TailLocalized) — wraps a [`BackingStore`]
///   in [`WriteMode::TailLocalizedWrites`].  Use for high-throughput data
///   ingestion where write ordering within the ring is acceptable.
///
/// # Examples
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use flush_buffer_ring::flush_behaviour::{QuickIO, WriteMode};
///
/// let file    = Arc::new(std::fs::File::open("/dev/null").unwrap());
/// let io_ring = Arc::new(parking_lot::Mutex::new(io_uring::IoUring::new(8).unwrap()));
///
/// let flusher = QuickIO::new(io_ring, file);
/// ```
pub enum QuickIO {
    /// Strictly serialised write appender (`IO_LINK` per SQE).
    Searalized(BackingStore),
    /// Tail-localised write appender (no ordering flags).
    TailLocalized(BackingStore),
}

impl QuickIO {
    /// Construct a [`QuickIO::Searalized`] from an existing ring and file handle.
    ///
    /// Writes submitted through this variant will use [`WriteMode::SerializedWrites`].
    pub fn link(io_uring: SharedAsyncFileWriter, file: Arc<File>) -> Self {
        QuickIO::Searalized(BackingStore::new(io_uring, file, WriteMode::SerializedWrites))
    }

    /// Construct a [`QuickIO::TailLocalized`] from an existing ring and file handle.
    ///
    /// Writes submitted through this variant will use [`WriteMode::TailLocalizedWrites`].
    pub fn new(io_uring: SharedAsyncFileWriter, file: Arc<File>) -> Self {
        QuickIO::TailLocalized(BackingStore::new(
            io_uring,
            file,
            WriteMode::TailLocalizedWrites,
        ))
    }

    /// Submit an **asynchronous** flush of the given buffer to its assigned LSS slot.
    ///
    /// Reads the buffer's data, offset, and user data via the [`FlushableBuffer`] trait
    /// and dispatches a fire-and-forget write SQE.
    ///
    /// Returns immediately; the caller must poll for completion.
    ///
    /// # Safety
    ///
    /// The buffer data must remain valid until the CQE is observed.
    pub fn submit_buffer<B: FlushableBuffer>(&self, buffer: &B) {
        let buffer_data = buffer.buffer_data();
        let at = buffer.offset();
        let user_data = buffer.user_data(); // In this case, the user data is always a buffer's pinned location in memmory
        let submit_entry = buffer.submit_entry();
        self.submit_buffer_raw(buffer_data, at, user_data, submit_entry);
    }

    /// Submit an **asynchronous** flush using raw parameters.
    ///
    /// This is the low-level method that takes individual parameters.
    /// Prefer [`submit_buffer`](Self::submit_buffer) when possible.
    ///
    /// Returns immediately; the caller must poll for completion.
    ///
    /// # Safety
    ///
    /// `buffer_data` must remain valid until the CQE is observed.
    pub fn submit_buffer_raw(&self, buffer_data: &[u8], at: u64, user_data: u64, submit_entry: &SubmitQueueEntry) {
        match self {
            QuickIO::Searalized(a) | QuickIO::TailLocalized(a) => {
                let _ = a.submit(buffer_data, at, user_data, submit_entry);
            }
        }
    }

    /// Submit an `fdatasync` with `IO_DRAIN` and block until it completes.
    ///
    /// `IO_DRAIN` causes the kernel to complete every previously submitted SQE
    /// before executing this one, so all in-flight `submit_buffer` writes are
    /// guaranteed durable before this returns.
    pub fn sync_data(&self) -> io::Result<()> {
        let backing_store = self.get_backing_store();

        let sqe = opcode::Fsync::new(types::Fd(backing_store.store.as_raw_fd()))
            .flags(types::FsyncFlags::DATASYNC)
            .build()
            .flags(squeue::Flags::IO_DRAIN);

        let mut ring = backing_store.flusher.lock();

        unsafe {
            ring.submission()
                .push(&sqe)
                .map_err(|_| io::Error::other("SQ full"))?;
        }

        ring.submit_and_wait(1)?;

        let cqe = ring
            .completion()
            .next()
            .ok_or_else(|| io::Error::other("no CQE for sync_data"))?;
        drop(ring);

        if cqe.result() < 0 {
            return Err(io::Error::from_raw_os_error(-cqe.result()));
        }

        Ok(())
    }

    /// Acquire exclusive access to the `io_uring` instance's completion queue.
    ///
    pub fn get_cqueue(
        &self,
    ) -> parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, IoUring> {
        match self {
            QuickIO::Searalized(appender) | QuickIO::TailLocalized(appender) => {
                appender.cqueue()
            }
        }
    }

    fn get_backing_store(&self) -> &BackingStore {
        match self {
            QuickIO::Searalized(backing_store) | QuickIO::TailLocalized(backing_store) => {
                backing_store
            }
        }
    }
}
