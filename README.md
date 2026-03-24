# Buffer Ring

A latch-free I/O buffer ring implementation for concurrent log-structured storage. Built on io_uring for efficient asynchronous I/O operations on Linux systems. Intended to be the sole write path for Bloom_lfs.

## Overview

This crate implements a fixed-size ring of 4-KILOBYTE-aligned buffers that amortizes individual I/O operations into larger, sequential writes before they are dispatched to stable storage. It provides latch-free concurrent access using a single packed atomic state word per buffer, making it suitable for high-throughput scenarios where multiple threads need to write to the same buffer simultaneously.

### Key Features

- **Latch-free writes**: No global locks; all state is managed through atomic operations
- **O_DIRECT compatible**: All buffers are 4-KILOBYTE aligned for direct I/O
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

For most applications, automatic flushing should be enabled by default:

```rust
use std::sync::Arc;
use buffer_ring::{BufferRing, BufferRingOptions, QuikIO};

// Create a ring with 4 buffers, 1 MB each, auto-flushing enabled
let options = BufferRingOptions {
    capacity: 4,
    buffer_size: buffer_ring::ONE_MEGABYTE_BLOCK,
    io_instance: Some(Arc::new(QuikIO::new(file))),
    auto_flush: true,
    auto_rotate: true,
};

let ring = BufferRing::with_options(options);
```

### Manual Flushing for Custom Protocols

If you need to implement custom buffer protocols or have specific flushing requirements, opt out of automatic flushing:

```rust
use std::sync::Arc;
use buffer_ring::{BufferRing, BufferRingOptions, QuikIO};

let flusher = Arc::new(QuikIO::new(file));

// Create a ring with MANUAL flushing
let options = BufferRingOptions {
    capacity: 4,
    buffer_size: buffer_ring::ONE_MEGABYTE_BLOCK,
    io_instance: Some(flusher),
    auto_flush: false,  // Disable automatic flushing
    auto_rotate: true,
};

let ring = BufferRing::with_options(options);

// Now you have full control over when buffers are flushed
```

### Key Methods for Manual Flushing

When `auto_flush` is `false`, use these methods to control flushing:

```rust
// Check if the current buffer is sealed (full)
let current = ring.current_buffer(Ordering::Acquire);
if current.is_sealed() {
    // Implement your custom flushing logic here
    ring.flush(&current);
}

// Or flush the current buffer explicitly at any time
let current = ring.current_buffer(Ordering::Acquire);
ring.flush(&current);

// Or flush a specific buffer
ring.flush(&my_buffer);
```

## Builder Configuration

`BufferRingOptions` provides configuration via struct fields:

```rust
let options = BufferRingOptions {
    capacity: 8,                      // Number of buffers in the ring
    buffer_size: buffer_ring::ONE_MEGABYTE_BLOCK,  // Size of each buffer
    io_instance: Some(flusher),       // I/O dispatcher
    auto_flush: true,                 // Enable/disable auto-flush
    auto_rotate: true,                // Enable/disable auto-rotate
};
```

### Configuration Details

| Field         | Type                  | Default | Description                                    |
| ------------- | --------------------- | ------- | ---------------------------------------------- |
| `capacity`    | `usize`               | 0       | Number of buffers in the ring                  |
| `buffer_size` | `usize`               | 0       | Size of each buffer (must be multiple of 4096) |
| `io_instance` | `Option<Arc<QuikIO>>` | None    | I/O dispatcher (test mode if None)             |
| `auto_flush`  | `bool`                | true    | Automatically flush when buffer sealed         |
| `auto_rotate` | `bool`                | true    | Automatically rotate to next buffer            |

> **Note**: Buffer size is fixed at multiples of 4096 for `O_DIRECT` compatibility. The constant `ONE_MEGABYTE_BLOCK` (1 MB) is recommended.

## When to Use Manual Flushing

Choose **manual flushing** (`auto_flush: false`) when:

- Implementing custom buffer protocols or serialization formats
- You need explicit control over flush timing for performance tuning
- You must batch multiple buffers before dispatching to storage
- Your workload has specific flush semantics beyond simple "on seal" behavior

Choose **automatic flushing** (`auto_flush: true`, the default) when:

- You want simplicity and predictable, automatic I/O dispatch
- Standard log-structured storage semantics are sufficient
- Thread safety and lock-free concurrency are your priorities

