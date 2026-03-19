# Buffer Ring

A latch-free I/O buffer ring implementation for concurrent log-structured storage. Built on io_uring for efficient asynchronous I/O operations on Linux systems. Intended to be the sole write path for Bloom_Db

## Overview

This crate implements a fixed-size ring of 1 MB-aligned buffers that amortizes individual I/O operations into larger, sequential writes before they are dispatched to stable storage. It provides latch-free concurrent access using a single packed atomic state word per buffer, making it suitable for high-throughput scenarios where multiple threads need to write to the same buffer simultaneously.

### Key Features

- **Latch-free writes**: No global locks; all state is managed through atomic operations
- **O_DIRECT compatible**: All buffers are 1 MB-aligned (ONE_MEGABYTE_BLOCK) for direct I/O
- **Concurrent amortization**: Multiple threads fill one buffer before flush
- **Flexible flushing**: Automatic or manual control over when buffers are dispatched
- **Ring-based rotation**: Seamlessly rotates to the next buffer when sealed

## System Architecture

### State Word Layout

All per-buffer metadata is packed into a single `AtomicUsize`, ensuring self-consistent snapshots:

```
┌────────────────┬────────────────┬──────────────────┬───────────────────┬──────────┐
│  Bits 63..32   │  Bits 31..8    │  Bits 7..2       │  Bit 1            │  Bit 0   │
│  write offset  │  writer count  │  (reserved)      │  flush-in-prog    │  sealed  │
└────────────────┴────────────────┴──────────────────┴───────────────────┴──────────┘
```

### Flush Protocol

The ring implements the flush protocol from the LLAMA paper without global locks:

1. **Identify** the page state to write
2. **Seize** space in the active buffer via atomic fetch-and-add
3. **Check** atomically whether reservation succeeded; if the buffer is full, seal and rotate
4. **Write** payload into reserved range while flush-in-progress bit prevents premature dispatch

## Usage

### Basic Setup with Automatic Flushing

For most applications, automatic flushing should probaly enabled by default:

```rust
use std::sync::Arc;
use flush_buffer_ring::{BufferRing, FlushRingOptions, QuickIO};

let flusher = Arc::new(QuickIO::with_no_wait_appender(io_uring, file));

// Create a ring with 4 buffers, 1 MB each, auto-flushing enabled
let ring = FlushRingOptions::new()
    .buffers(4)
    .flusher(flusher)
    .build();
```

### Manual Flushing for Custom Protocols

If you need to implement custom buffer protocols or have specific flushing requirements, **opt out of automatic flushing using `FlushRingOptions`**:

```rust
use std::sync::Arc;
use flush_buffer_ring::{BufferRing, FlushRingOptions, QuickIO};

let flusher = Arc::new(QuickIO::with_no_wait_appender(io_uring, file));

// Create a ring with MANUAL flushing
let ring = FlushRingOptions::new()
    .buffers(4)
    .auto_flush(false)  // ⚠️ Disable automatic flushing
    .flusher(flusher)
    .build();

// Now you have full control over when buffers are flushed
```

### Key Methods for Manual Flushing

When `auto_flush` is `false`, use these methods to control flushing:

```rust
// Check if the current buffer is sealed (full)
if ring.current_buffer_full() {
    // Implement your custom flushing logic here
    let buffer = ring.current_buffer();
    ring.flush(&buffer);
}

// Or flush the current buffer explicitly at any time
ring.flush_current_buffer();

// Or flush a specific buffer
ring.flush(&my_buffer);
```

## Builder Configuration

`FlushRingOptions` provides a fluent API for customization:

```rust
let ring = FlushRingOptions::new()
    .buffers(8)                      // Set number of buffers
    .auto_flush(true)                // Enable/disable auto-flush (default: true)
    .flusher(flusher_behavior)       // Set the flush dispatcher
    .build();
```

### Configuration Details

| Option          | Type           | Default | Description                             |
| --------------- | -------------- | ------- | --------------------------------------- |
| `buffers()`     | `usize`        | 4       | Number of buffers in the ring           |
| `auto_flush()`  | `bool`         | true    | Automatically flush when buffer sealed  |
| `flusher()`     | `Arc<QuickIO>` | None    | I/O dispatcher (test mode if None)      |
| **Buffer Size** | —              | 1 MB    | **Always `ONE_MEGABYTE_BLOCK`** (fixed) |

> **Note**: Buffer size is intentionally fixed at 1 MB for `O_DIRECT` compatibility and efficient page-aligned I/O. All buffers in the ring use this size.

## When to Use Manual Flushing

Choose **manual flushing** (`auto_flush(false)`) when:

- Implementing custom buffer protocols or serialization formats
- You need explicit control over flush timing for performance tuning
- You must batch multiple buffers before dispatching to storage
- Your workload has specific flush semantics beyond simple "on seal" behavior

Choose **automatic flushing** (`auto_flush(true)`, the default) when:

- You want simplicity and predictable, automatic I/O dispatch
- Standard log-structured storage semantics are sufficient
- Thread safety and lock-free concurrency are your priorities


## ⚠️ Critical Warnings: Manual Flushing Pitfalls

**When you disable automatic flushing, you assume significant responsibility for system correctness.** Below are the most dangerous scenarios:

### 1. **Ring Exhaustion (Deadlock)**

The most critical danger. If all buffers become sealed and none are flushed, the ring becomes **completely stuck**. New write attempts will fail with `BufferError::RingExhausted`.

```rust
// DANGEROUS: Auto-flush disabled, but never flush!
let ring = FlushRingOptions::new()
    .buffers(4)
    .auto_flush(false)  // ⚠️ Manual flush required
    .build();

// Write until all buffers seal...
// Now ring.put() returns RingExhausted on every thread!
// Application DEADLOCKED - cannot progress.
```

**Fix**: Establish a flush schedule. Every sealed buffer MUST eventually be flushed.

```rust
// CORRECT: Regular flushing prevents exhaustion
for batch in incoming_batches {
    // ... write data ...
    if ring.current_buffer_full() {
        ring.flush_current_buffer();  // ✓ Flush regularly
    }
}
```

### 2. **Premature Buffer Reuse (Data Corruption)**

If a buffer is reset before its I/O completes, new data can overwrite in-flight data:

```rust
// DANGEROUS: Reset before I/O completion
let buffer = ring.current_buffer();
ring.flush(buffer);
ring.reset_buffer(buffer);  // ⚠️ Too early! I/O not done!
// Now buffer can be reused and new data overwrites pending I/O
```

**Fix**: Only `reset_buffer()` after confirmed I/O completion:

```rust
// CORRECT: Reset after I/O completion handler
fn on_io_complete(buffer: &FlushBuffer) {
    // I/O is confirmed done
    ring.reset_buffer(buffer);  // ✓ Safe now
}
```

### 3. **Lost Writes and Data Races**

Without careful synchronization, concurrent threads can corrupt buffer state:

```rust
// DANGEROUS: Unsynchronized concurrent access
let buffer = ring.current_buffer();
buffer.increment_writers();    // Thread A
let _ = buffer.set_sealed_bit_true();  // Thread B
ring.flush(buffer);            // Both threads unsynchronized!
```

**Fix**: Use the `put()` method which handles synchronization internally, or implement your own CAS-based locking.

### 4. **Flushed Buffer Still Locked (Ring Stall)**

If a buffer is stuck in `flush-in-progress` state, it never re-enters the ring:

```rust
// DANGEROUS: Set flush-in-progress without resetting
let buffer = ring.current_buffer();
buffer.set_flush_in_progress();
// ... forget to call reset_buffer() ...
// This buffer is now **permanently locked**
// Ring slowly exhausts as buffers are permanently claimed
```

**Fix**: Always pair `flush()` with an eventual `reset_buffer()` call:

```rust
// CORRECT: Flush and reset in matching pair
ring.flush(buffer);
// ... I/O dispatcher receives callback ...
on_io_completion(buffer);
ring.reset_buffer(buffer);  // ✓ Re-enable for ring
```



### 5. **no Flusher Registered (Automatic Resets)**

When `auto_flush` is false and no `QuickIO` is registered, buffers reset immediately (test mode):

```rust
let ring = FlushRingOptions::new()
    .auto_flush(false)
    .flusher(None)  // No actual I/O dispatcher
    .build();

// Buffers reset immediately without actual I/O
// Data is lost! Designed for testing only.
```

**Fix**: Always register a `flusher` in production:

```rust
let flusher = Arc::new(QuickIO::with_wait_appender(io_uring, file));
let ring = FlushRingOptions::new()
    .auto_flush(false)
    .flusher(flusher)  // ✓ Real dispatcher
    .build();
```


### 6. **no new current Buffer set after seel**

When a caller seals a buffer, they must ensure that a new current buffer is set. They can do so manually through there own protocols or
throught the built in `rotate_after_seal()` method. The `rotate_after_seal()` method which rotates the ring away from the from its current state to the next available buffer.


``` rust
    let buffer = ring.current_buffer();
    let _ = buffer.set_sealed_bit_true(); 

    self.rotate_after_seal(buffer.pos)?; // Rotates to the next available buffer
    
    ring.flush(buffer);           

```



## Detailed Reference: Manual Flushing APIs

### Methods for Manual Control

#### `current_buffer() -> &'static FlushBuffer`

Get the active buffer for custom protocols:

```rust
let active = ring.current_buffer();
let state = active.state.load(Ordering::Acquire);
let offset = state_offset(state);  // Parse packed state
```

**Safety Notes:**
- The returned reference is only valid for the current rotation cycle
- Ring may rotate anytime if the current buffer is sealed
- Safe for read-only inspection only

#### `is_current_buffer_sealed() -> bool`

Check if current buffer is sealed (full):

```rust
if ring.is_current_buffer_sealed() {
    ring.flush_current_buffer();
}
```

**Use for:** Intelligent batching decisions

#### `flush_current_buffer()`

Convenience method to flush the active buffer:

```rust
let buffer = ring.current_buffer();
buffer.set_sealed_bit_true()?;
ring.flush_current_buffer();  // Dispatch to I/O
```

**Equivalent to:**
```rust
ring.flush(ring.current_buffer());
```

#### `flush(&buffer: &FlushBuffer)`

Explicit dispatch of a specific buffer:

```rust
ring.flush(buffer);  // Sets flush-in-progress bit
```

**Must be paired with:**
- `reset_buffer()` after I/O completion
- Monitoring via your `QuickIO` dispatcher

#### `reset_buffer(&buffer: &FlushBuffer)`

Clear state after I/O completion:

```rust
// Called from I/O completion handler
fn on_completion(buffer: &FlushBuffer) {
    ring.reset_buffer(buffer);  // Re-enable for ring
}
```

**Critical:** Do NOT call until I/O is confirmed complete.

### Complete Manual Flushing Protocol

```rust
use flush_buffer_ring::{FlushRingOptions, QuickIO};
use std::sync::Arc;

// 1. Create ring with manual control
let flusher = Arc::new(QuickIO::with_wait_appender(...));
let ring = Arc::new(
    FlushRingOptions::new()
        .buffers(8)
        .auto_flush(false)  // Enable manual mode
        .flusher(flusher)
        .build()
);

// 2. Register I/O completion callback (typically in io_uring code)
let ring_clone = Arc::clone(&ring);
async_register_completion_handler(move |buffer| {
    // I/O is done, disk is safe
    ring_clone.reset_buffer(buffer);
});

// 3. Main write loop with manual flush control
for entry in entries {
    loop {
        let current = ring.current_buffer();
        
        match current.reserve_space(entry.len()) {
            Ok(offset) => {
                current.write(offset, entry);
                current.decrement_writers();
                break;
            }
            Err(_) => {
                // Buffer full, must flush
                let _ = current.set_sealed_bit_true();
                ring.rotate_after_seal(current.pos); // Must rotate
                

                ring.flush_current_buffer();
            }
        }
    }
}



```
 If need be we can keep track of an address range slot for every buffer For the purpose of log Structured Systems, this is needed as buffers should never write to the same location twice. The `next_address_range` attribute may be atomically
 incremented to sort of logically move the the BufferRing along the log.

## Implementation Checklist for Manual Flushing

When implementing manual flushing, verify:

- [ ] Every sealed buffer is eventually flushed
- [ ] Current buffer has been set
- [ ] `reset_buffer()` is called only after I/O completion
- [ ] No buffer is preemptively reset before I/O starts
- [ ] `QuickIO` is registered (not None)
- [ ] Completion callbacks are properly synchronized
- [ ] Ring exhaustion is monitored and alerts configured
- [ ] Tests verify your flush schedule cannot deadlock
- [ ] Documentation explains custom flush semantics to users

## Flush Behaviors

The crate provides built-in flush strategies via `QuickIO`:

```rust
use flush_buffer_ring::QuickIO;
use std::sync::Arc;

// Parallel flushing: multiple buffers dispatched concurrently
let parallel = QuickIO::with_no_wait_appender(io_uring, file);

// Serial flushing: buffers dispatched one at a time
let serial = QuickIO::with_wait_appender(io_uring, file);
```

## Error Handling

The ring returns `BufferError` variants to indicate various conditions:

```rust
pub enum BufferError {
    InsufficientSpace,        // Buffer too full for this write
    EncounteredSealedBuffer,  // Buffer was sealed; retry with new one
    RingExhausted,            // All buffers busy; back off and retry
    FlushInProgress,          // Flush operation already in progress
    InvalidState,             // Internal state corrupted
}
```

## Thread Safety

The ring is fully thread-safe:

- All buffers can be accessed from multiple threads simultaneously
- No global locks; only atomic operations and CAS loops
- State is self-consistent within each atomic snapshot
- Gracefully handles concurrent sealing, rotation, and flushing

## Performance Characteristics

- **Write latency**: Sub-microsecond atomic operations (no locks)
- **Memory overhead**: Fixed ~64 bytes per buffer for metadata
- **I/O batching**: Amortizes overhead by buffering multiple writes per flush

## Examples

### Simple Concurrent Writes

```rust
use std::sync::Arc;
use std::thread;
use flush_buffer_ring::{BufferRing, FlushRingOptions, QuickIO};

let flusher = Arc::new(QuickIO::with_no_wait_appender(io_uring, file));
let ring = Arc::new(
    FlushRingOptions::new()
        .buffers(4)
        .flusher(flusher)
        .build()
);

let mut handles = vec![];

for _ in 0..4 {
    let r = ring.clone();
    handles.push(thread::spawn(move || {
        for i in 0..1000 {
            let payload = format!("entry_{}", i).as_bytes().to_vec();
            // Write will autoflushed when buffer is sealed
            // (Real usage would reserve space first)
        }
    }));
}

for handle in handles {
    handle.join().unwrap();
}
```

### Custom Flushing Logic

```rust
use flush_buffer_ring::{FlushBufferRing, FlushRingOptions};
use std::sync::Arc;

let ring = Arc::new(
    FlushRingOptions::new()
        .buffers(4)
        .auto_flush(false)  // Disable automatic flushing
        .flusher(flusher)
        .build()
);

// Custom flush strategy: flush every 5 buffers
let mut flush_count = 0;

// ... write operations ...

if ring.is_current_buffer_sealed() {
    ring.flush_current_buffer();
    flush_count += 1;
    
    if flush_count >= 5 {
        // Custom logic: wait for all flushes to complete, etc.
        flush_count = 0;
    }
}
```

## Constants

- `ONE_MEGABYTE_BLOCK = 1024 * 1024` (1 MB): Fixed buffer size for all rings

## Implementation Notes

- Buffers are allocated with `malloc` and manually aligned to `ONE_MEGABYTE_BLOCK`
- All state transitions use atomic compare-exchange loops
- The flush-in-progress bit prevents race conditions during I/O dispatch
- Ring rotation uses a simple index scanning strategy to find available buffers
- No memory barriers are used beyond those in atomic operations

## Testing

Run the comprehensive test suite:

```bash
cargo test
```

## License

license = "GPL-3.0"
