//! # Flush Buffer — Latch-Free I/O Buffer Ring
//!
//! This module implements LLAMA's in-memory write-staging layer: a fixed-size
//! ring of 4 KB-aligned [`FlushBuffer`]s that amortises individual page-state
//! writes into larger, sequential I/O operations before they are dispatched to
//! the [`LogStructuredStore`](crate::log_structured_store::LogStructuredStore).
//!
//! ## Design Goals
//!
//! | Goal                    | Mechanism                                                  |
//! |-------------------------|------------------------------------------------------------|
//! | Latch-free writes       | Single packed [`AtomicUsize`] state word per buffer        |
//! | `O_DIRECT` compatibility| 4 KB-aligned allocation via [`Buffer::new_aligned`]       |
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
//! 5. **On failure** at step 3, write a "Failed Flush" sentinel into the reserved
//!    space.  This wastes a few bytes but removes all ambiguity about which writes
//!    succeeded.
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

pub mod flush_buffer;
pub mod quik_io;
pub mod ring;
pub mod state;

pub use crate::quik_io::{FlushableBuffer, QuikIO, WriteMode};
pub use crate::ring::{BufferRing, BufferRingOptions};

pub use crate::state::State;

use std::{
    cell::UnsafeCell,
    sync::{Arc, atomic::AtomicUsize},
    usize,
};

use io_uring::squeue::Entry;

use std::alloc::{Layout, alloc_zeroed};

/// A 4 KB-aligned, heap-allocated byte buffer suitable for `O_DIRECT` I/O.
///
/// `Buffer` owns a single contiguous allocation that is aligned to a
/// FOUR_KB_BLOCK (4 096 bytes) — the minimum alignment required by
/// `O_DIRECT` on all common block devices.
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
    /// Total allocation size in bytes.
    size: AtomicUsize,
}

impl Buffer {
    /// Allocate a zeroed, [`FOUR_KB_BLOCK`]-aligned buffer of `size` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `size` is not a multiple of [`FOUR_KB_BLOCK`], if the layout
    /// is otherwise invalid, or if the allocator returns a null pointer.
    pub fn new_aligned(size: usize) -> Self {
        let layout = Layout::from_size_align(size, FOUR_KB_BLOCK).expect("invalid layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned allocation failed");

        Self {
            buffer: UnsafeCell::new(ptr),
            size: AtomicUsize::new(size),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(
            self.size.load(std::sync::atomic::Ordering::Acquire),
            FOUR_KB_BLOCK,
        )
        .unwrap();
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
const SEALED_BIT: usize = 1 << 0;

/// Bit 1 of the state word — set while a flush is in progress.
///
/// Prevents a second flush from being fired concurrently and prevents new
/// writers from entering a buffer that is already being drained.
pub const FLUSH_IN_PROGRESS_BIT: usize = 1 << 1;

/// Amount added to the state word to record one additional active writer.
const WRITER_SHIFT: usize = 8;
const WRITER_ONE: usize = 1 << WRITER_SHIFT;

/// Mask covering the writer-count field (bits 8..32).
const WRITER_MASK: usize = 0x00FF_FFFF00;

/// The write-offset field occupies the top 32 bits of the state word.
const OFFSET_SHIFT: usize = 32;

/// Amount added to the state word to advance the write offset by one byte.
const OFFSET_ONE: usize = 1 << OFFSET_SHIFT;

/// Default number of buffers in a [`FlushBufferRing`].
pub const RING_SIZE: usize = 4;

/// The size of a 1 MB page
pub const ONE_MEGABYTE_BLOCK: usize = 1024 * 1024;

// The size of a 1 KB page
pub const FOUR_KB_BLOCK: usize = 4096;

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
pub fn state_flush_in_progress(state: usize) -> bool {
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

/// A single 4 KB-aligned latch-free I/O buffer.
///
/// Multiple threads write into a `FlushBuffer` concurrently by atomically
/// claiming non-overlapping byte ranges through [`FlushBuffer::reserve_space`].  Once the
/// buffer is full (or explicitly sealed), it is dispatched to a [`QuikIO`] instance for
/// asynchronous flushes. It it subsequently reset for reuse
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
    pub(crate) state: AtomicUsize,

    /// Backing aligned byte store shared with the `io_uring` submission path.
    pub(crate) buf: SharedBuffer,

    /// Position of this buffer within the parent [`FlushBufferRing`].
    pub(crate) pos: usize,

    /// The LSS address slot assigned to this buffer at seal time.
    ///
    /// Assigned by [`FlushBufferRing::next_address_range`] via fetch-add;
    /// guaranteed unique across all concurrently sealed buffers.
    pub(crate) local_address: AtomicUsize,

    /// The most recently submitted `io_uring` SQE for this buffer.
    ///
    /// Stored so that a failed CQE can re-fire the exact same write without
    /// re-constructing the SQE.  Guarded by the flush-in-progress state
    /// transition — only one thread may write or read this field at a time.
    pub(crate) sqe: UnsafeCell<Option<Entry>>,
}

impl FlushBuffer {
    pub fn get_pos(&self) -> usize {
        self.pos
    }

    pub fn get_sqe(&self) -> Option<&Entry> {
        let sqe = unsafe { (*self.sqe.get()).as_ref() };
        sqe
    }
}

unsafe impl Send for FlushBuffer {}
unsafe impl Sync for FlushBuffer {}
