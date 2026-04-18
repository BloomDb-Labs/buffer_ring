use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
    },
};

use crate::{
    BufferError, BufferMsg, FLUSH_IN_PROGRESS_BIT, FOUR_KB_BLOCK, FlushBuffer, OFFSET_SHIFT,
    SEALED_BIT, quik_io::QuikIO, state_offset, state_sealed, state_writers,
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

    cq_tx: Option<Sender<(u64, usize)>>,
}

pub struct BufferRingOptions {
    capacity: usize,
    buffer_size: usize,
    io_instance: Option<Arc<QuikIO>>,
    auto_flush: bool,
    auto_rotate: bool,
    cq_tx: Option<Sender<(u64, usize)>>,
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
            cq_tx: None,
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
        self.buffer_size = size;
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

    /// Attach a completion channel.
    ///
    /// Returns a `Receiver` that yields `(file_offset, byte_count)` each time
    /// a buffer is confirmed written by a CQE. The sender end is stored in the
    /// ring; call `check_cque` to drive it.
    pub fn completion_receiver(&mut self) -> Receiver<(u64, usize)> {
        let (tx, rx) = mpsc::channel();
        self.cq_tx = Some(tx);
        rx
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
    pub fn with_options(options: &mut BufferRingOptions) -> BufferRing {
        let buffers: Vec<Arc<FlushBuffer>> = (0..options.capacity)
            .map(|i| Arc::new(FlushBuffer::new_buffer(i, options.buffer_size)))
            .collect();

        let buffers = Pin::new(buffers.into_boxed_slice());
        let current = &*buffers[0] as *const FlushBuffer as *mut FlushBuffer;

        let instance = options.io_instance.take();

        BufferRing {
            current_buffer: AtomicPtr::new(current),
            ring: buffers,
            next_index: AtomicUsize::new(1),
            size: options.capacity,
            next_address_range: AtomicUsize::new(0),
            store: instance,
            auto_flush: options.auto_flush,
            auto_rotate: options.auto_rotate,
            cq_tx: options.cq_tx.take(),
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

                let _ = store.wait_for_all();
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
        let current_size = current_ref.size();

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
                    Arc::as_ptr(new_buffer) as *mut FlushBuffer,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );

                self.__reserve_buf_addr(new_buffer, current_size);

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
        buffer.state.store(0, Ordering::SeqCst);
        buffer.local_address.store(0, Ordering::Release);
    }

    /// Internal completion queue processing.
    ///
    /// Drains all available CQEs and re-submits any failed writes
    pub fn check_cque(&self) -> Result<(), String> {
        let Some(store) = &self.store else {
            return Err("Store not present".to_string());
        };

        loop {
            let cqes = store.cqe();
            if cqes.is_empty() {
                return Ok(());
            }

            for cqe in cqes {
                if cqe.user_data() == 0 {
                    continue;
                }

                let ptr = cqe.user_data() as *const FlushBuffer;
                let buffer = unsafe { &*ptr };

                if cqe.result() < 0 {
                    // Retry failed write
                    if let Some(sqe) = unsafe { (*buffer.sqe.get()).as_ref() } {
                        let mut ring_guard = store.ring();
                        let _ = unsafe { ring_guard.submission().push(sqe) };
                        let _ = ring_guard.submit();
                    }
                } else {
                    // Success: record range + reset buffer
                    if let Some(tx) = &self.cq_tx {
                        let file_offset = buffer.local_address(Ordering::Acquire) as u64;
                        let byte_count = buffer.size();
                        let _ = tx.send((file_offset, byte_count));
                    }
                    self.reset_buffer(buffer); // <-- now reliable
                }
            }
        }
    }

    /// Seals and flushes the current buffer
    pub fn flush_current(&self) -> Result<(), BufferError> {
        let current_ptr = self.current_buffer.load(Ordering::Acquire);
        let current = unsafe { current_ptr.as_ref().ok_or(BufferError::InvalidState)? };

        if current.size() == 0 {
            return Ok(());
        }

        let _ = current.seal();

        // Reserve exact size at seal time (matches your updated RingWriter)
        let actual_len = current.size();
        let slot = self.incrment_address(actual_len, Ordering::SeqCst);
        current.local_address.store(slot, Ordering::Release);

        self.flush(current);
        let _ = self.rotate_after_seal(current.pos);

        Ok(())
    }

    /// Atomically loads the address range
    pub fn next_address(&self, ordering: Ordering) -> usize {
        self.next_address_range.load(ordering)
    }

    /// Atomically increments the lss address range of a flush buffer.
    pub fn incrment_address(&self, val: usize, ordering: Ordering) -> usize {
        self.next_address_range.fetch_add(val, ordering)
    }

    /// Reserve exact address space for a buffer at the moment it becomes active.
    /// Called when rotating to a new current buffer.
    fn __reserve_buf_addr(&self, buffer: &FlushBuffer, size: usize) {
        let slot = self.incrment_address(size, Ordering::SeqCst);
        let _ = buffer
            .local_address
            .compare_exchange(0, slot, Ordering::AcqRel, Ordering::Relaxed);
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
