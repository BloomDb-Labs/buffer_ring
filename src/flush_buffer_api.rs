use crate::*;

impl FlushBuffer {
    /// Create a new `FlushBuffer` at ring position `buffer_number` with a
    /// `size`-byte aligned backing allocation.
    ///
    /// The initial LSS address slot is set to `buffer_number` so that buffers
    /// are pre-assigned non-overlapping slots at construction time.  The ring
    /// will update this via [`set_address`](Self::set_address)
    /// each time the buffer is reused.
    pub fn new_buffer(buffer_number: usize, size: usize) -> FlushBuffer {
        Self {
            state: AtomicUsize::new(0),
            buf: Arc::new(Buffer::new_aligned(size)),
            pos: buffer_number,
            local_address: AtomicUsize::new(0),
            submit_queue_entry: UnsafeCell::new(None),
        }
    }

    pub fn numbered_buffer(buf_pos: usize) -> FlushBuffer {
        Self {
            state: AtomicUsize::new(0),
            buf: Arc::new(Buffer::new_aligned(ONE_MEGABYTE_BLOCK)),
            pos: buf_pos,
            local_address: AtomicUsize::new(0),
            submit_queue_entry: UnsafeCell::new(None),
        }
    }

    pub fn default() -> FlushBuffer {
        Self {
            state: AtomicUsize::new(0),
            buf: Arc::new(Buffer::new_aligned(ONE_MEGABYTE_BLOCK)),
            pos: 0,
            local_address: AtomicUsize::new(0),
            submit_queue_entry: UnsafeCell::new(None),
        }
    }

    /// Atomically update this buffer's LSS address slot to `address_space`.
    ///
    /// Returns `Ok(previous_slot)` on success or `Err(observed)` if the CAS
    /// fails (another thread updated the slot concurrently).
    pub fn set_address(&mut self, address_space: usize) -> Result<usize, usize> {
        let range = self.local_address.load(Ordering::Relaxed);
        self.local_address.compare_exchange(
            range,
            address_space,
            Ordering::Acquire,
            Ordering::Relaxed,
        )
    }

    /// Return `true` if this buffer is open to new writers.
    ///
    /// A buffer is available when neither the sealed bit nor the
    /// flush-in-progress bit is set.
    pub fn is_available(&self) -> bool {
        self.state.load(Ordering::Acquire) & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) == 0
    }

    /// Attempt to atomically reserve `payload_size` bytes in this buffer.
    ///
    /// On success returns the byte offset at which the caller should write its
    /// payload.  The caller **must** call [`decrement_writers`](Self::decrement_writers)
    /// once the write is complete.
    ///
    /// # Errors
    ///
    /// * [`BufferError::EncounteredSealedBuffer`] — the buffer is sealed or a
    ///   flush is in progress; the caller should ask the ring to rotate.
    /// * [`BufferError::InsufficientSpace`] — `payload_size` bytes would exceed
    ///   [`ONE_MEGABYTE_BLOCK`]; the caller should seal the buffer and retry on the
    ///   next one.
    /// * [`BufferError::FailedReservation`] — the CAS failed due to contention;
    ///   the caller should retry immediately.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `payload_size > ONE_MEGABYTE_BLOCK`.
    pub fn reserve_space(&self, payload_size: usize) -> Result<usize, BufferError> {
        assert!(
            payload_size <= ONE_MEGABYTE_BLOCK,
            "payload larger than buffer"
        );

        let state = self.state.load(Ordering::Acquire);

        if state & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) != 0 {
            return Err(BufferError::EncounteredSealedBuffer);
        }

        let offset = state_offset(state);

        if offset + payload_size > ONE_MEGABYTE_BLOCK {
            return Err(BufferError::InsufficientSpace);
        }

        // Analagous to the increment_writers() method
        let new = state
            .wrapping_add(payload_size * OFFSET_ONE)
            .wrapping_add(WRITER_ONE);

        match self
            .state
            .compare_exchange(state, new, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(offset),
            Err(_) => Err(BufferError::FailedReservation),
        }
    }

    /// Decrement the active-writer count by one.
    ///
    /// Should be called by every thread that previously succeeded at
    /// [`reserve_space`](Self::reserve_space) once it has finished copying its
    /// payload.  Returns the **previous** state word value.
    #[inline]
    pub fn decrement_writers(&self) -> usize {
        self.state.fetch_sub(WRITER_ONE, Ordering::AcqRel)
    }

    /// Increment the active-writer count by one.
    ///
    /// Returns the **previous** state word value.
    #[inline]
    pub fn increment_writers(&self) -> usize {
        self.state.fetch_add(WRITER_ONE, Ordering::AcqRel)
    }

    /// Set the flush-in-progress bit.
    ///
    /// Returns the **previous** state word value.  The caller should check
    /// whether the bit was already set in the returned value — only the thread
    /// that observes the bit transitioning from `0` to `1` owns the flush.
    #[inline]
    pub fn set_flush_in_progress(&self) -> usize {
        self.state.fetch_or(FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel)
    }

    /// Clear the flush-in-progress bit.
    ///
    /// Returns the **previous** state word value.
    #[inline]
    pub fn clear_flush_in_progress(&self) -> usize {
        self.state
            .fetch_and(!FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel)
    }

    /// Copy `payload` into the buffer at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must have obtained `offset` from a successful
    /// [`reserve_space`](Self::reserve_space) call and must not alias the same
    /// region from another thread.
    pub fn write(&self, offset: usize, payload: &[u8]) {
        debug_assert!(offset + payload.len() <= self.buf.size);

        unsafe {
            let dst = (*self.buf.buffer.get()).add(offset);
            std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, payload.len());
        }
    }

    /// Set the sealed bit, preventing any further reservations.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::EncounteredSealedBufferDuringCOMPEX`] if the
    /// buffer was already sealed before this call.
    pub fn set_sealed_bit_true(&self) -> Result<(), BufferError> {
        let prev = self.state.fetch_or(SEALED_BIT, Ordering::AcqRel);
        if state_sealed(prev) {
            Err(BufferError::EncounteredSealedBufferDuringCOMPEX)
        } else {
            Ok(())
        }
    }

    /// Clear the sealed bit, re-opening the buffer to new writers.
    ///
    /// Only succeeds when there are no active writers and no flush is in
    /// progress.
    ///
    /// # Errors
    ///
    /// * [`BufferError::ActiveUsers`] — writers or a flush are still active.
    /// * [`BufferError::EncounteredUnSealedBufferDuringCOMPEX`] — the buffer
    ///   was not sealed to begin with.
    /// * [`BufferError::FailedUnsealed`] — the CAS failed; retry.
    #[allow(unused)]
    pub(crate) fn set_sealed_bit_false(&self) -> Result<(), BufferError> {
        let current = self.state.load(Ordering::Acquire);

        if state_writers(current) != 0 || state_flush_in_progress(current) {
            return Err(BufferError::ActiveUsers);
        }

        if !state_sealed(current) {
            return Err(BufferError::EncounteredUnSealedBufferDuringCOMPEX);
        }

        match self.state.compare_exchange(
            current,
            current & !SEALED_BIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(_) => Err(BufferError::FailedUnsealed),
        }
    }

    /// Reset the write offset to zero, leaving all flag bits intact.
    ///
    /// Intended for use in tests only.  In production code the ring resets
    /// buffers through [`BufferRing::reset_buffer`].
    pub fn reset_offset(&self) {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let zeroed = current & 0x0000_0000_FFFF_FFFF;
            if self
                .state
                .compare_exchange(current, zeroed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Return a raw snapshot of the packed state word.
    ///
    /// Available in test builds only.  Use the `state_offset`, `state_writers`,
    /// `state_sealed`, and `state_flush_in_progress` helpers to decode the
    /// individual fields.
    ///
    //  This needs to be public
    #[cfg(test)]
    pub(crate) fn state_snapshot(&self) -> usize {
        use std::sync::atomic::Ordering;

        self.state.load(Ordering::Acquire)
    }

    /// Returns the current state of the buffer.
    pub fn state(&self) -> crate::State {
        crate::State::from(self.state.load(Ordering::Acquire))
    }

    /// Returns the local LSS address slot assigned to this buffer.
    pub fn local_address(&self) -> usize {
        self.local_address.load(Ordering::Acquire)
    }

    /// Returns a reference to the submit queue entry storage.
    pub fn sqe(&self) -> &UnsafeCell<Option<Entry>> {
        &self.submit_queue_entry
    }

    /// Returns the position of this buffer within the parent BufferRing.
    pub fn buffer_position(&self) -> usize {
        self.pos
    }
}
