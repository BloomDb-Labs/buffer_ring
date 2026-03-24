use std::{
    fs::OpenOptions,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    sync::{Arc, atomic::Ordering},
};

use buffer_ring::{
    BufferRing, BufferRingOptions, ONE_MEGABYTE_BLOCK, QuikIO, RING_SIZE, WriteMode, quik_io,
};

fn main() {
    let path = "test_store/simple_async.db";
    let (store, io) = open(path, WriteMode::SerializedWrites).expect("Got Store");

    let data = [5u8; 4096];

    let buffer = store.current_buffer(Ordering::Acquire);

    let _ = buffer.set_address(0).unwrap(); // or use the ring's incrment_address

    // Use increment_offset + write as before
    let offset = buffer
        .increment_offset(data.len())
        .expect("increment_offset failed");

    buffer.write(offset, &data);

    // Now the buffer knows its file offset
    io.submit_buffer(buffer);

    // // Drain completions (good practice)
    let _ = io.wait_for_all();

    let mut read_buffer = vec![0u8; 4096];
    io.read(read_buffer.as_mut_ptr(), 4096, 0).unwrap(); // offset 0 if you used slot 0

    assert_eq!(&read_buffer, &data);
}

fn open(
    path: impl AsRef<Path>,
    write_mode: WriteMode,
) -> Result<(BufferRing, Arc<QuikIO>), Box<dyn std::error::Error>> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = Arc::new(
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // O_DIRECT bypasses the kernel page cache.
            // INVARIANT: every buffer passed to read/write must be aligned to
            // FOUR_KB_PAGE — upheld by Buffer::new_aligned in flush_buffer.rs.
            .custom_flags(libc::O_DIRECT)
            .open(path.as_ref())?,
    );

    let flusher = Arc::new(match write_mode {
        WriteMode::TailLocalizedWrites => QuikIO::new(file.clone()),
        WriteMode::SerializedWrites => QuikIO::link(file.clone()),
    });

    let mut options = BufferRingOptions::new();
    options
        .auto_flush(false)
        .auto_rotate(false)
        .buffer_size(ONE_MEGABYTE_BLOCK)
        .capacity(RING_SIZE)
        .io_instance(flusher.clone());

    let ring = BufferRing::with_options(&mut options);

    Ok((ring, flusher.clone()))
}
