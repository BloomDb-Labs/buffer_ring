use crate::{FLUSH_IN_PROGRESS_BIT, OFFSET_SHIFT, SEALED_BIT, WRITER_MASK, WRITER_SHIFT};

/// A wrapper around the packed state word of a FlushBuffer.
///
/// The state word contains the write offset, writer count, and flag bits
/// packed into a single usize for atomic operations.
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub(crate) inner: usize,
}

impl State {
    /// Extracts the current offset out of the state variable
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.inner >> OFFSET_SHIFT
    }

    /// Extracts the current number of writers out of the state variable
    #[inline(always)]
    pub fn n_writers(&self) -> usize {
        (self.inner & WRITER_MASK) >> WRITER_SHIFT
    }

    /// Returns the sealed bit of the state variable
    #[inline(always)]
    pub fn sealed(&self) -> bool {
        self.inner & SEALED_BIT != 0
    }

    /// Returns the flush in progress bit of the state variable
    #[inline(always)]
    pub fn flushing(&self) -> bool {
        self.inner & FLUSH_IN_PROGRESS_BIT != 0
    }
}

impl From<usize> for State {
    fn from(inner: usize) -> Self {
        Self { inner }
    }
}

impl From<State> for usize {
    fn from(state: State) -> usize {
        state.inner
    }
}