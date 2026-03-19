#[cfg(feature = "loom")]
mod loom_tests {
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::sync::Arc;
    use loom::thread;

    // Minimal loom model of the last-writer protocol 
    //
    // We extract just the state-word logic from FlushBuffer and replay it
    // under loom's scheduler. This is the minimal reproduction of the
    // seal/flush/reset cycle that is stalling in production.
    //
    // State word layout mirrors Sflush_buffer.rs exactly:
    //   bits 63..32 = offset
    //   bits 31..8  = writers
    //   bit 1       = flush-in-progress
    //   bit 0       = sealed

    const SEALED_BIT: usize = 1 << 0;
    const FLUSH_IN_PROGRESS_BIT: usize = 1 << 1;
    const WRITER_SHIFT: usize = 8;
    const WRITER_ONE: usize = 1 << WRITER_SHIFT;
    const WRITER_MASK: usize = 0x00FF_FFFF00;
    const OFFSET_SHIFT: usize = 32;
    const OFFSET_ONE: usize = 1 << OFFSET_SHIFT;

    fn state_writers(state: usize) -> usize {
        (state & WRITER_MASK) >> WRITER_SHIFT
    }

    fn state_sealed(state: usize) -> bool {
        state & SEALED_BIT != 0
    }

    impl MockBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            state: AtomicUsize::new(0),
            capacity,
        }
    }

    fn try_start_flush(&self) -> bool {
        let prev = self
            .state
            .fetch_or(FLUSH_IN_PROGRESS_BIT, Ordering::AcqRel);

        (prev & FLUSH_IN_PROGRESS_BIT) == 0
    }

    // Returns Ok(offset) or Err — mirrors reserve_space
    fn reserve(&self, size: usize) -> Result<usize, &'static str> {
        loop {
            let s = self.state.load(Ordering::Acquire);

            if s & (SEALED_BIT | FLUSH_IN_PROGRESS_BIT) != 0 {
                return Err("sealed");
            }

            let offset = s >> OFFSET_SHIFT;

            if offset + size > self.capacity {
                return Err("insufficient");
            }

            let new = s
                .wrapping_add(size * OFFSET_ONE)
                .wrapping_add(WRITER_ONE);

            match self.state.compare_exchange(
                s,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(offset),
                Err(_) => continue,
            }
        }
    }

    // Writer finishes and may trigger flush
    fn finish_write(&self) -> bool {
        let prev = self.state.fetch_sub(WRITER_ONE, Ordering::AcqRel);

        let was_last = state_writers(prev) == 1;
        let was_sealed = state_sealed(prev);

        if was_last && was_sealed {
            return self.try_start_flush();
        }

        false
    }

    // Sealer path
    fn seal(&self) -> Result<bool, &'static str> {
        let prev = self.state.fetch_or(SEALED_BIT, Ordering::AcqRel);

        if state_sealed(prev) {
            return Err("already sealed");
        }

        let current = self.state.load(Ordering::Acquire);
        let writers_now = state_writers(current);

        if writers_now == 0 {
            return Ok(self.try_start_flush());
        }

        Ok(false)
    }

    fn reset(&self) {
        loop {
            let s = self.state.load(Ordering::Acquire);

            let reset =
                s & !(SEALED_BIT | FLUSH_IN_PROGRESS_BIT | (usize::MAX << OFFSET_SHIFT));

            if self
                .state
                .compare_exchange(
                    s,
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
    // ── Test 1: one writer, one sealer ────────────────────────────────────────
    //
    // Thread A reserves space successfully and is mid-write.
    // Thread B gets InsufficientSpace and seals.
    // Exactly one of them must trigger flush. Neither must be missed.
    //
    // This is the minimal reproduction of the stall scenario.
    #[test]
    fn loom_one_writer_one_sealer() {
        loom::model(|| {
            // Buffer capacity = 8 bytes. Thread A writes 6, Thread B tries 4
            // (insufficient) and seals.
            let buf = Arc::new(MockBuffer::new(8));
            let flushed = Arc::new(AtomicUsize::new(0));

            let buf_a = Arc::clone(&buf);
            let flushed_a = Arc::clone(&flushed);

            // Thread A: reserve 6 bytes, finish write
            let thread_a = thread::spawn(move || {
                match buf_a.reserve(6) {
                    Ok(_offset) => {
                        // Simulate write work — loom will interleave here
                        if buf_a.finish_write() {
                            flushed_a.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(_) => {} // buffer already sealed before we got in
                }
            });

            let buf_b = Arc::clone(&buf);
            let flushed_b = Arc::clone(&flushed);

            // Thread B: try to reserve 4 bytes, get InsufficientSpace, seal
            let thread_b = thread::spawn(move || {
                match buf_b.reserve(4) {
                    Err("insufficient") => {
                        match buf_b.seal() {
                            Ok(should_flush) => {
                                if should_flush {
                                    flushed_b.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            Err(_) => {} // another thread sealed first
                        }
                    }
                    Ok(_) => {} // squeezed in before seal
                    Err(_) => {} // already sealed
                }
            });

            thread_a.join().unwrap();
            thread_b.join().unwrap();

            // INVARIANT: flush must have been triggered exactly once
            // If this assert fires, loom will show you the exact interleaving
            let flush_count = flushed.load(Ordering::Relaxed);
            assert_eq!(
                flush_count, 1,
                "flush must be triggered exactly once — got {flush_count}"
            );

            // Reset for next loom iteration
            buf.reset();
        });
    }

    // ── Test 2: two writers, one sealer ───────────────────────────────────────
    //
    // Thread A and B both reserve successfully.
    // Thread C gets InsufficientSpace and seals.
    // The last of A/B to finish must trigger flush.
    // C must not trigger flush (writers still active when it seals).
    #[test]
    fn loom_two_writers_one_sealer() {
        loom::model(|| {
            let buf = Arc::new(MockBuffer::new(12));
            let flushed = Arc::new(AtomicUsize::new(0));

            // Thread A: reserve 4, finish
            let buf_a = Arc::clone(&buf);
            let flushed_a = Arc::clone(&flushed);
            let thread_a = thread::spawn(move || {
                if let Ok(_) = buf_a.reserve(4) {
                    if buf_a.finish_write() {
                        flushed_a.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            // Thread B: reserve 4, finish
            let buf_b = Arc::clone(&buf);
            let flushed_b = Arc::clone(&flushed);
            let thread_b = thread::spawn(move || {
                if let Ok(_) = buf_b.reserve(4) {
                    if buf_b.finish_write() {
                        flushed_b.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            // Thread C: try reserve 8 (insufficient), seal
            let buf_c = Arc::clone(&buf);
            let flushed_c = Arc::clone(&flushed);
            let thread_c = thread::spawn(move || {
                match buf_c.reserve(8) {
                    Err("insufficient") => {
                        if let Ok(should_flush) = buf_c.seal() {
                            if should_flush {
                                flushed_c.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    Ok(_) => {
                        if buf_c.finish_write() {
                            flushed_c.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(_) => {}
                }
            });

            thread_a.join().unwrap();
            thread_b.join().unwrap();
            thread_c.join().unwrap();

            let flush_count = flushed.load(Ordering::Relaxed);
            assert_eq!(
                flush_count, 1,
                "flush must be triggered exactly once — got {flush_count}"
            );

            buf.reset();
        });
    }

    // ── Test 3: writer decrements before sealer checks ────────────────────────
    //
    // The specific race we suspect:
    // Thread A reserves, Thread B seals, Thread A decrements — all interleaved.
    // After Thread A decrements, writers == 0 and buffer is sealed.
    // Somebody must notice and flush. Nobody must flush twice.
    #[test]
    fn loom_writer_decrements_before_sealer_checks() {
        loom::model(|| {
            let buf = Arc::new(MockBuffer::new(8));
            let flushed = Arc::new(AtomicUsize::new(0));

            let buf_a = Arc::clone(&buf);
            let flushed_a = Arc::clone(&flushed);

            // Thread A: reserve small amount, then finish (decrement)
            let thread_a = thread::spawn(move || {
                if let Ok(_) = buf_a.reserve(4) {
                    // Loom will try scheduling the decrement before and after
                    // Thread B's seal — this is the critical interleaving
                    if buf_a.finish_write() {
                        flushed_a.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            let buf_b = Arc::clone(&buf);
            let flushed_b = Arc::clone(&flushed);

            // Thread B: insufficient space, seals, checks writers
            let thread_b = thread::spawn(move || {
                match buf_b.reserve(6) {
                    Err("insufficient") => {
                        if let Ok(should_flush) = buf_b.seal() {
                            if should_flush {
                                flushed_b.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    Ok(_) => {
                        if buf_b.finish_write() {
                            flushed_b.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(_) => {}
                }
            });

            thread_a.join().unwrap();
            thread_b.join().unwrap();

            let flush_count = flushed.load(Ordering::Relaxed);
            assert_eq!(
                flush_count, 1,
                "flush must be triggered exactly once — got {flush_count}"
            );

            buf.reset();
        });
    }
}