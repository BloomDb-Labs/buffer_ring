use buffer_ring::{
    BufferError, BufferMsg, BufferRing, BufferRingOptions, ONE_MEGABYTE_BLOCK, QuikIO,
};
use flexi_logger::{FileSpec, Logger};
use log::debug;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _logger = Logger::try_with_str("debug")
        .unwrap()
        .log_to_file(FileSpec::default().directory("logs"))
        .start()
        .unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let file = Arc::new(temp_file.as_file().try_clone().unwrap());
    let io_dispatcher = Arc::new(QuikIO::link(Arc::clone(&file)));

    let mut options = BufferRingOptions::new();

    let completions = options.completion_receiver();

    options
        .capacity(4)
        .buffer_size(ONE_MEGABYTE_BLOCK)
        .io_instance(Arc::clone(&io_dispatcher))
        .auto_flush(true)
        .auto_rotate(true);

    let ring = Arc::new(BufferRing::with_options(&mut options));

    println!("Starting logging example with 5 threads (each with 30_000-message address range)...");
    let start = Instant::now();
    let mut handles = vec![];

    for thread_id in 0..5 {
        let ring_clone = Arc::clone(&ring);
        let handle = thread::spawn(move || {
            let mut local_count = 0;

            for i in 0..30_000 {
                let message_num = i + (30_000 * thread_id);

                let message = format!(
                    "[{:02}] [Thread-{}] Message {:03} \n",
                    thread_id, thread_id, message_num
                );

                if log_message(&ring_clone, message.as_bytes()).is_ok() {
                    local_count += 1;
                }
            }
            local_count
        });
        handles.push(handle);
    }

    let mut total_messages = 0;
    for handle in handles {
        total_messages += handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    println!(
        "Logging complete! {} messages in {:.2}s ({:.0}/s)",
        total_messages,
        elapsed.as_secs_f64(),
        total_messages as f64 / elapsed.as_secs_f64(),
    );

    let _ = ring.flush_current();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let _ = ring.check_cque();

    io_dispatcher.sync_data()?;

    // Give extra time for completions
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Drain completion queue thoroughly

    let _ = ring.check_cque();

    read_completed_ranges(&io_dispatcher, completions)?;
    drop(_logger);

    Ok(())
}


fn read_completed_ranges(
    io: &QuikIO,
    completions: std::sync::mpsc::Receiver<(u64, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut ranges: Vec<(u64, usize)> = completions.try_iter().collect();
    ranges.sort_by_key(|&(offset, _)| offset);

    debug!("\n--- {} completed flush range(s) ---", ranges.len());

    for (i, (file_offset, byte_count)) in ranges.iter().enumerate() {
        let mut buf = vec![0u8; *byte_count];

        // Reads the entire chunk of written data all at once
        io.read(buf.as_mut_ptr(), *byte_count, *file_offset)?;
        let text = String::from_utf8_lossy(&buf);

        debug!("[range {i}] offset={file_offset} bytes={byte_count}");
        println!("[range {i}] offset={file_offset} bytes={byte_count}");
        debug!("{}", text);
    }

    Ok(())
}

fn log_message(ring: &BufferRing, payload: &[u8]) -> Result<BufferMsg, BufferError> {
    loop {
        let current = ring.current_buffer(Ordering::Acquire);
        let reserve_result = current.reserve_space(payload.len());

        match &reserve_result {
            Err(BufferError::FailedReservation) | Err(BufferError::EncounteredSealedBuffer) => {
                continue;
            }
            _ => {}
        }

        match ring.put(current, reserve_result, payload) {
            Ok(msg) => return Ok(msg),
            Err(BufferError::EncounteredSealedBuffer) | Err(BufferError::RingExhausted) => {
                let _ = ring.check_cque();
                std::thread::yield_now();
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
