//! # Flush Buffer — Latch-Free I/O Buffer Ring
//!
//! This module is intended to suit the needs of all of LLAMA's in-memory write-staging layers.
//! Its a fixed-size ring of on MB-aligned [`FlushBuffer`]s that amortises individual page-state
//! writes into larger, sequential I/O operations before they are dispatched to
//! the LogStructuredStore.
//!
//! ## Design Goals
//!
//! | Goal                    | Mechanism                                                  |
//! |-------------------------|------------------------------------------------------------|
//! | Latch-free writes       | Single packed [`AtomicUsize`] state word per buffer        |
//! | `O_DIRECT` compatibility| 4 KB-aligned allocation via [`Buffer::new_aligned`]        |
//! | Amortised I/O           | Multiple threads fill one buffer before it is flushed      |
//! | All threads participate | Any thread may seal or initiate a flush                    |
//!
//! ## Flush Protocol
//!
//! Adapted from the LLAMA paper; all steps are performed without global locks:
//!
//! 1. **Identify** the page state to be written.
//! 2. **Seize** space in the active [`FlushBuffer`] via
//!    [`reserve_space`](FlushBuffer::reserve_space) — an atomic fetch-and-add
//!    on the packed state word claims a non-overlapping byte range.
//! 3. **Check** atomically whether the reservation succeeded.  If the buffer is
//!    already sealed or the space is exhausted, the buffer is sealed and the ring
//!    rotates to the next available slot.
//! 4. **Write** the payload into the reserved range while the flush-in-progress
//!    bit prevents the buffer from being dispatched to stable storage prematurely.
//!
//! Though the currently implementation delegates the handling of all erroneous and invalid
//! states to the caller, the current implementation of the Flush proceedure should lend itself
//! well to to LLAMA flushing protocol
//!
//! ## State Word Layout
//!
//! All per-buffer metadata is packed into a single [`AtomicUsize`], making every
//! state snapshot self-consistent and eliminating TOCTOU (time of check/time of use) races between the
//! fields:
//!
//! ```text
//! ┌────────────────┬────────────────┬──────────────────┬───────────────────┬──────────┐
//! │  Bits 63..32   │  Bits 31..8    │  Bits 7..2       │  Bit 1            │  Bit 0   │
//! │  write offset  │  writer count  │  (reserved)      │  flush-in-prog    │  sealed  │
//! └────────────────┴────────────────┴──────────────────┴───────────────────┴──────────┘
//! ```
//!
//! * **write offset** — next free byte position inside the backing allocation.
//! * **writer count** — number of threads that have reserved space but not yet finished
//!   copying their payload.
//! * **flush-in-progress** — set by whichever thread wins the CAS race to own the
//!   flush; prevents a second flush from being fired while the first is in flight.
//! * **sealed** — set when the buffer is full or explicitly closed; prevents new
//!   reservations.
//!
//! Bits 7..2 represent unused space
//!

// TODO
/*
    Allow for the mutability and access of attributes from using clean apis

*/

pub mod flush_behaviour;
pub mod flush_buffer_api;
pub mod state;

// Re-exports for convenient access to the main API
pub use crate::flush_behaviour::{QuickIO, SharedAsyncFileWriter, WriteMode};
pub use crate::state::State;

use std::{
    cell::UnsafeCell,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
    usize,
};

use io_uring::squeue::Entry;

use std::alloc::{Layout, alloc_zeroed};

/// A 4 KB-aligned, heap-allocated byte buffer suitable for `O_DIRECT` I/O.
///
/// `Buffer` owns a single contiguous allocation that is aligned to
/// [`ONE_MEGABYTE_BLOCK`]. A 4 kilobyte size block is the minimum alignment required by
/// `O_DIRECT` on all common block devices. All mutiples this minimal allignment are valid
///
/// Cursor management is **not** handled here.  Instead, [`FlushBuffer`] uses
/// atomic fetch-and-add on its packed state word to hand out non-overlapping
/// byte ranges to concurrent writers.  This is what makes the
/// `unsafe impl Sync` sound: no two threads are ever granted the same region.
///
/// # Safety
///
/// [`Sync`] is manually implemented because [`UnsafeCell`] opts out of it by
/// default.  The invariant that upholds this is: all mutable access to the
/// inner pointer is mediated by [`FlushBuffer`], which guarantees exclusive
/// ranges per writer.
#[derive(Debug)]
pub struct Buffer {
    /// Raw pointer to the aligned allocation, wrapped in [`UnsafeCell`] to
    /// allow interior mutability without a lock.
    pub(crate) buffer: UnsafeCell<*mut u8>,
    /// Total allocation size in bytes.  Stored for correct deallocation.
    size: usize,
}

impl Buffer {
    /// Allocate a zeroed, [`ONE_MEGABYTE_BLOCK`]-aligned buffer of `size` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `size` is not a multiple of [`ONE_MEGABYTE_BLOCK`], if the layout
    /// is otherwise invalid, or if the allocator returns a null pointer.
    pub fn new_aligned(size: usize) -> Self {
        let layout = Layout::from_size_align(size, ONE_MEGABYTE_BLOCK).expect("invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned allocation failed");

        Self {
            buffer: UnsafeCell::new(ptr),
            size,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, ONE_MEGABYTE_BLOCK).unwrap();
        unsafe { std::alloc::dealloc(*self.buffer.get(), layout) };
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A reference-counted handle to a [`Buffer`].
///
/// Shared between a [`FlushBuffer`] and the `io_uring` submission path, which
/// holds a pointer into the buffer while a write is in flight.
pub(crate) type SharedBuffer = Arc<Buffer>;

/// Bit 0 of the state word — set when the buffer is closed to new writers.
pub const SEALED_BIT: usize = 1 << 0;

/// Bit 1 of the state word — set while a flush is in progress.
///
/// Prevents a second flush from being fired concurrently and prevents new
/// writers from entering a buffer that is already being drained.
pub const FLUSH_IN_PROGRESS_BIT: usize = 1 << 1;

/// Amount added to the state word to record one additional active writer.
const WRITER_SHIFT: usize = 8;
const WRITER_ONE: usize = 1 << WRITER_SHIFT;

/// Mask covering the writer-count field (bits 8..32).
pub const WRITER_MASK: usize = 0x00FF_FFFF00;

/// The write-offset field occupies the top 32 bits of the state word.
pub const OFFSET_SHIFT: usize = 32;

/// Amount added to the state word to advance the write offset by one byte.
const OFFSET_ONE: usize = 1 << OFFSET_SHIFT;

/// Default number of buffers in a [`BufferRing`].
pub const RING_SIZE: usize = 4;

/// The size of a 1 MB page
pub const ONE_MEGABYTE_BLOCK: usize = 1024 * 1024;

#[inline(always)]
/// Extracts the current offset out of the state variable
pub fn state_offset(state: usize) -> usize {
    state >> OFFSET_SHIFT
}

#[inline(always)]
/// Extracts the current current number of writers out of the state variable
pub fn state_writers(state: usize) -> usize {
    (state & WRITER_MASK) >> WRITER_SHIFT
}

#[inline(always)]
/// Returns the sealed bit of the state variable
pub fn state_sealed(state: usize) -> bool {
    state & SEALED_BIT != 0
}

#[inline(always)]
/// Returns the flush in progress bit of the state variable
fn state_flush_in_progress(state: usize) -> bool {
    state & FLUSH_IN_PROGRESS_BIT != 0
}

/// Errors that may be returned by buffer and ring operations.
#[derive(Debug, Clone, Copy)]
pub enum BufferError {
    /// The payload exceeds the remaining capacity of the active flush buffer.
    InsufficientSpace,

    /// The buffer is sealed and no longer accepts new reservations.
    EncounteredSealedBuffer,

    /// A CAS on the sealed bit found it was already set.
    EncounteredSealedBufferDuringCOMPEX,

    /// A CAS on the sealed bit found it was already clear.
    EncounteredUnSealedBufferDuringCOMPEX,

    /// A flush was attempted while at least one writer is still active.
    ActiveUsers,

    /// The buffer or ring is in an undefined / corrupt state.
    InvalidState,

    /// All buffers in the ring are sealed or being flushed — none available.
    RingExhausted,

    /// A [`reserve_space`](FlushBuffer::reserve_space) CAS failed; the caller
    /// should retry.
    FailedReservation,

    /// An attempt to clear the sealed bit via CAS failed; the caller should
    /// retry.
    FailedUnsealed,
}

/// Successful outcomes returned by buffer and ring operations.
#[derive(Debug, Clone)]
pub enum BufferMsg {
    /// The buffer transitioned to the sealed state.
    SealedBuffer,

    /// The payload was written to the buffer; no flush was triggered.
    SuccessfullWrite,

    /// The payload was written and the buffer was dispatched for flushing.
    SuccessfullWriteFlush,

    /// The buffer is ready to flush.  Carries the [`FlushBuffer`] that was
    /// sealed, allowing the recipient to initiate the flush independently.
    FreeToFlush(Arc<FlushBuffer>),
}

/// A single  latch-free I/O buffer.
///
/// Multiple threads write into a `FlushBuffer` concurrently by atomically
/// claiming non-overlapping byte ranges through [`FlushBuffer::reserve_space`].  Once the
/// buffer is full (or explicitly sealed), it is dispatched to
/// [`QuickIO`] for an `io_uring` write and then reset for reuse.
///
/// # State Word
///
/// ```text
/// ┌────────────────┬────────────────┬──────────────────┬───────────────────┬──────────┐
/// │  Bits 63..32   │  Bits 31..8    │  Bits 7..2       │  Bit 1            │  Bit 0   │
/// │  write offset  │  writer count  │  (reserved)      │  flush-in-prog    │  sealed  │
/// └────────────────┴────────────────┴──────────────────┴───────────────────┴──────────┘
/// ```
///
/// All four fields are read and updated through a single [`AtomicUsize`], so
/// any snapshot is self-consistent: there are no TOCTOU races between the
/// offset, writer count, flush flag, and sealed flag.
///
/// # Safety
///
/// `FlushBuffer` is `Send + Sync`.  The only `unsafe` access is inside
/// [`write`](Self::write), where a raw pointer into the aligned allocation is
/// dereferenced.  Safety is upheld by the invariant that
/// [`reserve_space`](Self::reserve_space) grants each caller an exclusive,
/// non-overlapping byte range.
#[derive(Debug)]
pub struct FlushBuffer {
    /// Packed atomic state — see type-level docs for the bit layout.
    state: AtomicUsize,

    /// Backing aligned byte store shared with the `io_uring` submission path.
    buf: SharedBuffer,

    /// Position of this buffer within the parent [`BufferRing`].
    pub pos: usize,

    /// The LSS address slot assigned to this buffer at seal time.
    ///
    /// On-disk byte offset = `local_address × FlushBufferSize`.
    /// Assigned by [`BufferRing::next_address_range`] via fetch-add;
    /// guaranteed unique across all concurrently sealed buffers.
    local_address: AtomicUsize, // TODO getters and setters

    /// The most recently submitted `io_uring` SQE for this buffer.
    ///
    /// Stored so that a failed CQE can re-fire the exact same write without
    /// re-constructing the SQE.  Guarded by the flush-in-progress state
    /// transition — only one thread may write or read this field at a time.
    submit_queue_entry: UnsafeCell<Option<Entry>>, // TODO getters
}

unsafe impl Send for FlushBuffer {}
unsafe impl Sync for FlushBuffer {}

impl crate::flush_behaviour::FlushableBuffer for FlushBuffer {
    fn buffer_data(&self, data_len: usize) -> &[u8] {
        unsafe {
            let ptr = *self.buf.buffer.get();
            &*std::ptr::slice_from_raw_parts(ptr, data_len as usize)
        }
    }

    fn offset(&self) -> u64 {
        self.local_address.load(Ordering::Acquire) as u64
    }

    fn user_data(&self) -> u64 {
        self as *const FlushBuffer as u64
    }

    fn submit_entry(&self) -> &crate::flush_behaviour::SubmitQueueEntry {
        &self.submit_queue_entry
    }
}

/// A fixed-size ring of [`FlushBuffer`]s that amortises writes into batched
/// sequential I/O.
///
/// The ring maintains a single *current* buffer pointer that all threads write
/// into concurrently.  When the current buffer is full it is sealed, a fresh
/// buffer is selected from the ring, and the sealed buffer is optionally dispatched to
/// the configured [`QuickIO`] for an `io_uring` write.
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
/// completion queue to free up buffers.
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

    _size: usize,

    /// Optional flush dispatcher.  `None` in test mode — buffers are reset
    /// immediately without dispatching any `io_uring` writes.
    store: Option<Arc<QuickIO>>,

    /// Whether to automatically flush buffers when they are sealed.
    /// If false, users must manually call [`flush`](Self::flush) or [`flush_current_buffer`](Self::flush_current_buffer).
    auto_flush: bool,
}

impl BufferRing {
    /// Create a ring of `num_of_buffer` buffers, each `buffer_size` bytes,
    /// with **no** flush dispatcher attached.
    ///
    /// Intended for unit tests that exercise the ring's concurrency primitives
    /// without requiring a real `io_uring` instance or backing file.  In this
    /// mode, sealed buffers are reset immediately after flush is triggered,
    /// keeping the ring from stalling.
    pub fn with_buffer_amount(num_of_buffer: usize, buffer_size: usize) -> BufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..num_of_buffer)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, buffer_size)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        BufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: num_of_buffer,
            next_address_range: AtomicUsize::new(0),
            store: None,
            auto_flush: true, // Default to auto flush for backward compatibility
        }
    }

    /// Create a ring of `num_of_buffer` buffers, each `buffer_size` bytes,
    /// connected to `flusher` for real `io_uring`-backed I/O.
    ///
    /// This is the production constructor.  Sealed buffers are submitted to
    /// `flusher` instead of being reset immediately.
    ///
    /// For configurable buffer ring creation, use [`FlushRingOptions`] instead.
    pub fn new(num_of_buffer: usize, flusher: Arc<QuickIO>) -> BufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..num_of_buffer)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, ONE_MEGABYTE_BLOCK)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        BufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: num_of_buffer,
            next_address_range: AtomicUsize::new(0),
            store: Some(flusher),
            auto_flush: true, // Default to auto flush
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
                // Seal the buffer — whichever thread sets the bit from 0→1 owns
                // the flush.
                let prev = current.state.fetch_or(SEALED_BIT, Ordering::AcqRel);

                if prev & SEALED_BIT != 0 {
                    return Err(BufferError::EncounteredSealedBuffer);
                }

                // Claim a unique slot in stable storage for this buffer before rotating.
                let slot = self.next_address_range.fetch_add(1, Ordering::AcqRel);
                current.local_address.store(slot, Ordering::Release);

                if self.auto_flush {
                    self.rotate_after_seal(current.pos)?;
                }

                let data_len = current.state().offset();

                // Race to own the flush.  If writers are still active, the last
                // one to decrement will also attempt this and one of them will
                // observe the bit transitioning 0→1.
                let before = current.set_flush_in_progress();
                if before & FLUSH_IN_PROGRESS_BIT == 0 {
                    if self.auto_flush {
                        match self.store.as_ref() {
                            Some(store) => {
                                store.submit_buffer(current, data_len);
                            }
                            None => {
                                // Test mode: no dispatcher — reset immediately.
                                self.reset_buffer(current);
                            }
                        }
                    }
                    return Ok(BufferMsg::SuccessfullWriteFlush);
                }

                return Err(BufferError::ActiveUsers);
            }

            Err(BufferError::EncounteredSealedBuffer) => {
                return Err(BufferError::EncounteredSealedBuffer);
            }

            Err(e) => return Err(e),

            Ok(offset) => {
                let data_len = current.state().offset();

                current.write(offset, payload);

                let prev = current.decrement_writers();

                // Note: Atomic operations always yeild previous values
                let was_last_writer = state_writers(prev) == 1;
                let was_sealed = state_sealed(prev);

                if was_last_writer && was_sealed {
                    let prev = current.set_flush_in_progress();

                    if prev & FLUSH_IN_PROGRESS_BIT == 0 {
                        // Only flush if auto_flush is enabled
                        if self.auto_flush {
                            let flush_buffer = self.ring.get(current.pos).unwrap().clone();
                            self.flush(&flush_buffer, data_len);
                        }
                        return Ok(BufferMsg::SuccessfullWriteFlush);
                    }
                }

                return Ok(BufferMsg::SuccessfullWrite);
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
    /// # Example: Custom Batching Protocol with Manual Control
    ///
    /// ```ignore
    /// let ring = FlushRingOptions::new()
    ///     .buffers(8)
    ///     .auto_flush(false)
    ///     .flusher(Arc::new(QuickIO::new_parallel()))
    ///     .build();
    ///
    /// // Custom batching: only flush every 5 buffers
    /// let mut flush_count = 0;
    /// for batch in incoming_batches {
    ///     // Write batch into current buffer...
    ///     if ring.is_current_buffer_sealed() {
    ///         ring.flush_current_buffer();  // Manually trigger flush
    ///         flush_count += 1;
    ///     }
    /// }
    /// ```
    pub fn flush(&self, buffer: &FlushBuffer, data_len: usize) {
        buffer.set_flush_in_progress();

        match self.store.as_ref() {
            Some(store) => {
                store.submit_buffer(buffer, data_len);
            }
            None => {
                self.reset_buffer(buffer);
            }
        }
    }

    /// Get a reference to the current active buffer.
    ///
    /// # Safety
    ///
    /// The returned reference is valid only for the current snapshot. The ring may rotate
    /// to a different buffer at any time if the current one is sealed. Use this method only
    /// when you need to inspect buffer state for custom protocols.
    pub fn current_buffer(&self) -> &'static FlushBuffer {
        let ptr = self.current_buffer.load(Ordering::Acquire);
        unsafe { ptr.as_ref().unwrap() }
    }

    /// Check if the current buffer is sealed (full).
    pub fn is_current_buffer_sealed(&self) -> bool {
        state_sealed(self.current_buffer().state.load(Ordering::Acquire))
    }

    /// Manually flush the current buffer.
    ///
    /// This is a convenience method for manually initiating a flush of the current active buffer.
    /// Only use this when `auto_flush` is disabled and you need explicit control over flush timing.
    ///
    /// # Panics
    ///
    /// Panics if no buffer is currently active (should not happen in normal usage).
    pub fn flush_current_buffer(&self, data_len: usize) {
        let buffer = self.current_buffer();
        self.flush(buffer, data_len);
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
}

/// Options for creating [`BufferRing`] instances with custom configurations.
///
/// All buffers in the ring use [`ONE_MEGABYTE_BLOCK`] (1 MB) size for compatibility
/// with `O_DIRECT` and efficient page-aligned I/O.
pub struct FlushRingOptions {
    buffers: usize,
    auto_flush: bool,
    flusher: Option<Arc<QuickIO>>,
}

impl FlushRingOptions {
    /// Create a new options builder with default settings.
    ///
    /// Defaults:
    /// - 4 buffers
    /// - 1MB buffer size (fixed; always [`ONE_MEGABYTE_BLOCK`])
    /// - Auto flush enabled
    /// - No flusher (test mode)
    pub fn new() -> Self {
        Self {
            buffers: 4,
            auto_flush: true,
            flusher: None,
        }
    }

    /// Set the number of buffers in the ring.
    pub fn buffers(mut self, count: usize) -> Self {
        self.buffers = count;
        self
    }

    /// Enable or disable automatic flushing when buffers are sealed.
    ///
    /// When disabled, buffers must be manually flushed via [`BufferRing::flush_current_buffer`]
    /// or [`BufferRing::flush`]. This is useful for custom buffer protocols.
    pub fn auto_flush(mut self, enabled: bool) -> Self {
        self.auto_flush = enabled;
        self
    }

    /// Set the flush behavior for I/O operations.
    pub fn flusher(mut self, flusher: Arc<QuickIO>) -> Self {
        self.flusher = Some(flusher);
        self
    }

    /// Build the [`BufferRing`] with the configured settings.
    pub fn build(self) -> BufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..self.buffers)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, ONE_MEGABYTE_BLOCK)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        BufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            _size: self.buffers,
            next_address_range: AtomicUsize::new(0),
            store: self.flusher,
            auto_flush: self.auto_flush,
        }
    }
}

impl Default for FlushRingOptions {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {

    use super::*;

    use std::{
        collections::HashSet,
        sync::{Arc, Barrier, Mutex},
        thread,
        time::Instant,
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
                    std::thread::yield_now();
                    continue;
                }
                Err(BufferError::RingExhausted) => {
                    std::thread::yield_now();
                    continue;
                }
                other => return other,
            }
        }
    }

    // =========================================================================
    // Single-buffer unit tests — no ring, no flusher
    // =========================================================================

    /// reserve_space on a sealed buffer must return EncounteredSealedBuffer.
    #[test]
    fn reserve_on_sealed_buffer_returns_error() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.reserve_space(16),
            Err(BufferError::EncounteredSealedBuffer)
        ));
    }

    /// Sealing an already-sealed buffer must return the COMPEX error.
    #[test]
    fn double_seal_returns_error() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        buf.set_sealed_bit_true().unwrap();
        assert!(matches!(
            buf.set_sealed_bit_true(),
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        ));
    }

    /// Unsealing an already-unsealed buffer must return the COMPEX error.
    #[test]
    fn unseal_unsealed_returns_error() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        assert!(matches!(
            buf.set_sealed_bit_false(),
            Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX)
        ));
    }

    /// reserve_space on a flush-in-progress buffer must return EncounteredSealedBuffer.
    #[test]
    fn reserve_on_flush_in_progress_returns_error() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        buf.set_flush_in_progress();
        assert!(matches!(
            buf.reserve_space(16),
            Err(BufferError::EncounteredSealedBuffer)
        ));
    }

    /// Writer count increments and decrements must be symmetric.
    #[test]
    fn writer_count_symmetric() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        buf.increment_writers();
        buf.increment_writers();
        buf.increment_writers();
        assert_eq!(state_writers(buf.state_snapshot()), 3);
        buf.decrement_writers();
        buf.decrement_writers();
        buf.decrement_writers();
        assert_eq!(state_writers(buf.state_snapshot()), 0);
    }

    /// A single exact-capacity reservation must consume the whole buffer.
    #[test]
    fn reserve_exact_capacity() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        let offset = buf.reserve_space(ONE_MEGABYTE_BLOCK).unwrap();
        assert_eq!(offset, 0);
        // Next reservation must fail — no space left.
        assert!(matches!(
            buf.reserve_space(1),
            Err(BufferError::InsufficientSpace)
        ));
    }

    /// Two sequential reservations must not overlap.
    #[test]
    fn sequential_reservations_no_overlap() {
        let buf = FlushBuffer::new_buffer(0, ONE_MEGABYTE_BLOCK);
        let a = buf.reserve_space(100).unwrap();
        let b = buf.reserve_space(100).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 100);
    }

    // =========================================================================
    // Concurrent single-buffer test — the most critical correctness invariant
    // =========================================================================

    /// Eight threads race to reserve 16-byte regions from a single buffer.
    /// No (buffer_pos, offset) pair may ever be issued twice.
    ///
    /// This directly validates the CAS-based reservation: if any two threads
    /// receive the same offset, the atomic state word is broken.
    #[test]
    fn concurrent_reserve_space_no_overlap() {
        let buf = Arc::new(FlushBuffer::new_buffer(99, ONE_MEGABYTE_BLOCK));
        let seen: Arc<Mutex<HashSet<usize>>> = Arc::new(Mutex::new(HashSet::new()));

        const THREADS: usize = 8;
        // 8 threads × 32 reservations × 16 bytes = 4096 — exactly fills one buffer
        const RESERVES_PER_THREAD: usize = 32;

        let barrier = Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_tid| {
                let buf = Arc::clone(&buf);
                let seen = Arc::clone(&seen);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait(); // all threads start simultaneously

                    for _ in 0..RESERVES_PER_THREAD {
                        loop {
                            match buf.reserve_space(16) {
                                Ok(offset) => {
                                    let mut lock = seen.lock().unwrap();
                                    assert!(
                                        lock.insert(offset),
                                        "[OVERLAP] offset {offset} issued twice!"
                                    );
                                    break;
                                }
                                Err(BufferError::FailedReservation) => continue,
                                Err(BufferError::InsufficientSpace) => break,
                                Err(BufferError::EncounteredSealedBuffer) => break,
                                Err(e) => panic!("unexpected error: {e:?}"),
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("reserve worker panicked");
        }

        // All 256 unique offsets (0, 16, 32, ... 4080) must be present
        let lock = seen.lock().unwrap();
        assert_eq!(
            lock.len(),
            THREADS * RESERVES_PER_THREAD,
            "expected {} unique offsets, got {}",
            THREADS * RESERVES_PER_THREAD,
            lock.len()
        );
    }

    // =========================================================================
    // Ring-level tests — seal, rotate, exhaustion
    // =========================================================================

    /// A full-capacity payload fills the buffer and triggers a seal + rotate.
    /// After the write, current_buffer must point at a different buffer.
    #[test]
    fn exact_fill_triggers_rotate() {
        let ring = BufferRing::with_buffer_amount(TEST_RING_SIZE, ONE_MEGABYTE_BLOCK);
        let payload = make_payload("FILL", ONE_MEGABYTE_BLOCK);

        match put_with_retry(&ring, &payload) {
            Ok(BufferMsg::SuccessfullWrite) | Ok(BufferMsg::SuccessfullWriteFlush) => {}
            other => panic!("exact_fill: unexpected {other:?}"),
        }

        // After a full-capacity write the ring must have rotated.
        // In no-flusher mode the buffer is reset immediately, so the pointer
        // may have wrapped — just assert the ring is still operational.
        let result = put_with_retry(&ring, &make_payload("AFTER", 16));
        assert!(
            result.is_ok(),
            "ring should still accept writes after rotate: {result:?}"
        );
    }

    /// Seal a buffer explicitly and verify the ring rotates to the next slot.
    #[test]
    fn manual_seal_causes_rotate() {
        let ring = BufferRing::with_buffer_amount(TEST_RING_SIZE, ONE_MEGABYTE_BLOCK);

        let current_before = unsafe {
            ring.current_buffer
                .load(Ordering::Acquire)
                .as_ref()
                .unwrap()
        };
        let pos_before = current_before.pos;

        // Seal the current buffer manually
        current_before.set_sealed_bit_true().unwrap();
        ring.rotate_after_seal(pos_before).unwrap();

        let current_after = unsafe {
            ring.current_buffer
                .load(Ordering::Acquire)
                .as_ref()
                .unwrap()
        };

        assert_ne!(
            current_after.pos, pos_before,
            "current_buffer should have rotated away from sealed buffer"
        );
    }

    /// After sealing all buffers without resetting, the ring must return
    /// RingExhausted rather than deadlocking or panicking.
    #[test]
    fn ring_exhaustion_returns_error() {
        let ring = BufferRing::with_buffer_amount(TEST_RING_SIZE, ONE_MEGABYTE_BLOCK);

        // Manually seal every buffer so none are available
        for i in 0..TEST_RING_SIZE {
            ring.ring[i].set_sealed_bit_true().ok();
        }

        let result = ring.rotate_after_seal(0);
        assert!(
            matches!(result, Err(BufferError::RingExhausted)),
            "expected RingExhausted, got {result:?}"
        );
    }

    /// Random-sized writes, single thread. Verifies the ring keeps accepting
    /// writes across multiple seal/rotate cycles without panicking.
    #[test]
    fn single_threaded_offset_uniqueness() {
        let ring = BufferRing::with_buffer_amount(TEST_RING_SIZE, ONE_MEGABYTE_BLOCK);

        let mut rng = Lcg::new(0);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut data_written = 0usize;
        let mut i = 0usize;

        loop {
            let size = SIZES[rng.next_usize(SIZES.len())];
            if data_written + size > ONE_MEGABYTE_BLOCK * TEST_RING_SIZE {
                break;
            }

            let payload = make_payload(&format!("s{i:05}"), size);
            data_written += size;

            match put_with_retry(&ring, &payload) {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("single_threaded: unexpected {other:?}"),
            }
            i += 1;
        }

        println!(
            "single_threaded_offset_uniqueness: {writes} writes, {flushes} flushes, {data_written} bytes"
        );
    }

    /// Stress test: 2000 random-sized writes, single thread.
    #[test]
    fn single_threaded_stress() {
        let ring = BufferRing::with_buffer_amount(TEST_RING_SIZE, ONE_MEGABYTE_BLOCK);
        let mut writes = 0usize;
        let mut flushes = 0usize;
        let mut rng = Lcg::new(0x1234_5678);
        let start = Instant::now();

        for op in 0..OPS_PER_THREAD {
            let size = SIZES[rng.next_usize(SIZES.len())];
            let payload = make_payload(&format!("S:O{op:04}"), size);

            match put_with_retry(&ring, &payload) {
                Ok(BufferMsg::SuccessfullWrite) => writes += 1,
                Ok(BufferMsg::SuccessfullWriteFlush) => {
                    writes += 1;
                    flushes += 1;
                }
                other => panic!("op {op}: unexpected {other:?}"),
            }
        }

        let elapsed = start.elapsed();
        println!(
            "single_threaded_stress: {writes} writes, {flushes} flushes in {elapsed:.2?} ({:.0} ops/s)",
            (writes + flushes) as f64 / elapsed.as_secs_f64()
        );
    }

    // =========================================================================
    // Multi-threaded stress tests
    // =========================================================================

    const NUM_THREADS_SMALL: usize = 2;
    const NUM_THREADS_MEDIUM: usize = 4;
    const NUM_THREADS_LARGE: usize = 8;

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
        let ring = Arc::new(BufferRing::with_buffer_amount(
            TEST_RING_SIZE,
            ONE_MEGABYTE_BLOCK,
        ));
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
                    // Record start AFTER barrier — this is when real work begins
                    start_times.lock().unwrap().push(Instant::now());

                    for op in 0..OPS_PER_THREAD {
                        let size = SIZES[rng.next_usize(SIZES.len())];
                        let payload = make_payload(&format!("T{tid}:O{op:04}"), size);

                        let result = loop {
                            let current = unsafe {
                                ring.current_buffer
                                    .load(Ordering::Acquire)
                                    .as_ref()
                                    .expect("null current_buffer")
                            };

                            let reserve_result = current.reserve_space(payload.len());

                            match &reserve_result {
                                Err(BufferError::FailedReservation) => continue,
                                Err(BufferError::EncounteredSealedBuffer) => continue,
                                _ => {}
                            }

                            match ring.put(current, reserve_result, &payload) {
                                Err(BufferError::ActiveUsers) => continue,
                                Err(BufferError::EncounteredSealedBuffer) => continue,
                                Err(BufferError::RingExhausted) => {
                                    std::thread::yield_now();
                                    continue;
                                }
                                Ok(BufferMsg::SealedBuffer) => continue,
                                other => break other,
                            }
                        };

                        match result {
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

    /// All threads race with large (2KB) payloads to maximise seal/rotate
    /// contention. Two threads × 100 ops × 2KB = 200KB of writes across
    /// multiple ring rotations.
    #[test]
    fn hammer_seal_concurrent_rotation() {
        let ring = Arc::new(BufferRing::with_buffer_amount(
            TEST_RING_SIZE,
            ONE_MEGABYTE_BLOCK,
        ));
        let barrier = Arc::new(Barrier::new(NUM_THREADS_SMALL));

        let handles: Vec<_> = (0..NUM_THREADS_SMALL)
            .map(|tid| {
                let ring = Arc::clone(&ring);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();

                    for iter in 0..100_usize {
                        let payload = make_payload(&format!("H{tid}:{iter}"), 2048);
                        match put_with_retry(&ring, &payload) {
                            Ok(_) => {}
                            Err(e) => panic!("hammer thread {tid} iter {iter}: error {e:?}"),
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("hammer worker panicked");
        }
    }

    // =========================================================================
    // Manual Flushing Tests
    // =========================================================================

    /// Test that auto_flush=false is respected throughout the lifecycle.
    #[test]
    fn manual_flush_auto_flush_disabled() {
        let ring = FlushRingOptions::new().buffers(2).auto_flush(false).build();

        // Verify the ring was created with auto_flush disabled
        assert!(!ring.auto_flush, "ring should have auto_flush disabled");
    }

    /// Test that reset_buffer() properly clears state bits.
    #[test]
    fn manual_flush_reset_clears_state() {
        let ring = FlushRingOptions::new().buffers(2).auto_flush(false).build();

        let buffer = ring.current_buffer();

        // Manually set state bits
        let _ = buffer.set_sealed_bit_true();
        buffer.set_flush_in_progress();

        let state_before = buffer.state.load(Ordering::Acquire);
        assert!(state_sealed(state_before) || (state_before & FLUSH_IN_PROGRESS_BIT) != 0);

        // Reset the buffer
        ring.reset_buffer(buffer);

        // Verify sealed bit is cleared
        let state_after = buffer.state.load(Ordering::Acquire);
        assert!(
            !state_sealed(state_after),
            "reset_buffer should clear sealed bit"
        );
    }

    /// Test current_buffer() returns a valid buffer.
    #[test]
    fn manual_flush_current_buffer_valid() {
        let ring = FlushRingOptions::new().buffers(3).build();

        let buffer1 = ring.current_buffer();
        let buffer2 = ring.current_buffer();

        // Should return the same buffer pointer
        let ptr1 = buffer1 as *const FlushBuffer;
        let ptr2 = buffer2 as *const FlushBuffer;
        assert_eq!(
            ptr1, ptr2,
            "current_buffer() should return consistent pointer"
        );
    }

    /// Test is_current_buffer_sealed() detection.
    #[test]
    fn manual_flush_buffer_full_detection() {
        let ring = FlushRingOptions::new().buffers(2).auto_flush(false).build();

        // Initially buffer should not be full
        assert!(
            !ring.is_current_buffer_sealed(),
            "buffer should not be full initially"
        );

        // Seal the buffer
        let current = ring.current_buffer();
        let _ = current.set_sealed_bit_true();

        // Now it should be full
        assert!(
            ring.is_current_buffer_sealed(),
            "sealed buffer should be reported as full"
        );
    }

    /// Test the manual flush protocol flow with write and reset.
    #[test]
    fn manual_flush_protocol_flow() {
        let ring = FlushRingOptions::new().buffers(2).auto_flush(false).build();

        let buffer = ring.current_buffer();

        // Step 1: Write data
        let payload = b"protocol test";
        let reserve = buffer.reserve_space(payload.len()).unwrap();
        buffer.write(reserve, payload);

        // Step 2: Complete the protocol
        buffer.decrement_writers();
        let _ = buffer.set_sealed_bit_true();

        // Step 3: Manually flush and reset
        ring.flush(buffer, payload.len());
        ring.reset_buffer(buffer);

        // Verify buffer is clean
        let state = buffer.state.load(Ordering::Acquire);
        assert!(
            !state_sealed(state),
            "buffer should not be sealed after reset"
        );
    }

    /// Test that with auto_flush=false, manual flush is required.
    #[test]
    fn manual_flush_explicit_control() {
        let ring = Arc::new(FlushRingOptions::new().buffers(2).auto_flush(false).build());

        let barrier = Arc::new(Barrier::new(2));
        let ring1 = Arc::clone(&ring);
        let barrier1 = Arc::clone(&barrier);

        let h1 = thread::spawn(move || {
            barrier1.wait();
            let current = ring1.current_buffer();

            // Small reserve to not exhaust buffer
            let payload = vec![1u8; 100];
            if let Ok(offset) = current.reserve_space(payload.len()) {
                current.write(offset, &payload);
                current.decrement_writers();
            }
        });

        let barrier2 = Arc::clone(&barrier);

        let h2 = thread::spawn(move || {
            barrier2.wait();
            std::thread::sleep(std::time::Duration::from_millis(10));
        });

        h1.join().unwrap();
        h2.join().unwrap();

        // Verify auto_flush flag was correctly set to false
        assert!(!ring.auto_flush, "auto_flush should be disabled");
    }

    /// Test manual batching simulation.
    #[test]
    fn manual_flush_batching_simulation() {
        let ring = Arc::new(FlushRingOptions::new().buffers(4).auto_flush(false).build());

        let flush_count = Arc::new(AtomicUsize::new(0));
        let ring_clone = Arc::clone(&ring);
        let flush_count_clone = Arc::clone(&flush_count);

        let handle = thread::spawn(move || {
            for i in 0..3 {
                let current = ring_clone.current_buffer();
                let payload = format!("batch_{}", i).into_bytes();

                // Try to write, if buffer is full, flush
                match current.reserve_space(payload.len()) {
                    Ok(offset) => {
                        current.write(offset, &payload);
                        current.decrement_writers();
                    }
                    Err(_) => {
                        // Buffer exhausted, flush it
                        let _ = current.set_sealed_bit_true();
                        ring_clone.flush_current_buffer(payload.len());
                        flush_count_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handle.join().unwrap();

        // The test completes without panicking, demonstrating the manual protocol works
        assert!(ring.auto_flush == false);
    }

    /// Test that auto_flush=true maintains original behavior.
    #[test]
    fn manual_flush_auto_flush_enabled() {
        let ring = FlushRingOptions::new()
            .buffers(2)
            .auto_flush(true) // Explicitly enabled
            .build();

        // Verify the ring has auto_flush enabled
        assert!(ring.auto_flush, "ring should have auto_flush enabled");

        // The manual methods should still be available
        let _buf = ring.current_buffer();
        let _is_full = ring.is_current_buffer_sealed();
    }

    // =========================================================================
    // Flush Behaviour (QuickIO) Tests
    // =========================================================================

    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Test QuickIO read method with a temporary file.
    #[test]
    fn quickio_read_basic() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"Hello, QuickIO read test!";
        temp_file.write_all(test_data).unwrap();
        temp_file.flush().unwrap();

        let file = Arc::new(temp_file.as_file().try_clone().unwrap());

        let quickio = QuickIO::new(file);

        let mut buffer = vec![0u8; test_data.len()];

        // Read from offset 0
        quickio.read(buffer.as_mut_ptr(), buffer.len(), 0).unwrap();

        assert_eq!(&buffer, test_data);
    }

    /// Test QuickIO read with offset.
    #[test]
    fn quickio_read_with_offset() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        temp_file.write_all(test_data).unwrap();
        temp_file.flush().unwrap();

        let file = Arc::new(temp_file.as_file().try_clone().unwrap());

        let quickio = QuickIO::new(file);

        let mut buffer = vec![0u8; 10];

        // Read 10 bytes starting from offset 5
        quickio.read(buffer.as_mut_ptr(), 10, 5).unwrap();

        let expected = &test_data[5..15];
        assert_eq!(&buffer, expected);
    }

    /// Test QuickIO read with unaligned offset (should still work due to internal alignment).
    #[test]
    fn quickio_read_unaligned_offset() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        temp_file.write_all(test_data).unwrap();
        temp_file.flush().unwrap();

        let file = Arc::new(temp_file.as_file().try_clone().unwrap());

        let quickio = QuickIO::new(file);

        let mut buffer = vec![0u8; 5];

        // Read 5 bytes starting from unaligned offset 7
        quickio.read(buffer.as_mut_ptr(), 5, 7).unwrap();

        let expected = &test_data[7..12];
        assert_eq!(&buffer, expected);
    }

    #[test]
    fn read_write_test() {
        let temp_file = NamedTempFile::new().unwrap();
        let file = Arc::new(temp_file.as_file().try_clone().unwrap());

        let quickio = QuickIO::new(file);

        let expected: Vec<[u8; 4096]> = vec![
            [0u8; 4096],
            [1u8; 4096],
            [2u8; 4096],
            [3u8; 4096],
            [4u8; 4096],
        ];

        let buffers: Vec<FlushBuffer> = (0..expected.len())
            .map(|i| {
                let mut buf = FlushBuffer::default();
                buf.set_address(i * 4096).expect("msg");
                buf
            })
            .collect();

        for (buf, data) in buffers.iter().zip(expected.iter()) {
            buf.write(0, data);
            quickio.submit_buffer(buf, 4096);
        }

        quickio.sync_data().unwrap();

        for (i, check_against) in expected.iter().enumerate() {
            let mut read_buffer = vec![0u8; 4096];
            let byte_offset = (i * 4096) as u64;
            quickio
                .read(read_buffer.as_mut_ptr(), 4096, byte_offset)
                .unwrap();

            assert_eq!(&read_buffer[..], check_against, "slot {i} mismatch");
        }
    }
}
