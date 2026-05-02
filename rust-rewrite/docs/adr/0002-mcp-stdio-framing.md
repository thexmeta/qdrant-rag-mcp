# ADR 0002: MCP Stdio Framing and Concurrency

## Status
Accepted

## Context
The Model Context Protocol (MCP) server communicates with clients (like Claude Code) over standard input/output (`stdio`) using JSON-RPC messages. In an asynchronous Rust environment (using `tokio`), naive implementations of reading from `stdin` (e.g., manually reading until a newline delimiter) can easily lead to buffer deadlocks, dropped messages, or thread starvation.

Furthermore, processing incoming requests can involve CPU-bound tasks (like heavy AST parsing or local model inference). If these tasks run directly on `tokio`'s async worker threads, they will block the event loop and starve the I/O readers.

## Decision
We will enforce strict protocol framing at the `tokio` layer to guarantee non-blocking I/O and decouple the transport layer from the processing logic.

### Implementation Details
1. **Transport Layer Framing:** We will use `tokio_util::codec::LinesCodec` wrapping `tokio::io::stdin()` and `tokio::io::stdout()`. This guarantees that partial JSON payloads are safely buffered without blocking the async runtime, and correctly handles message boundaries (newlines).
2. **Payload Limits:** We will configure `LinesCodec` with a conservative maximum payload limit (e.g., 10MB) to prevent memory exhaustion attacks or unhandled massive file context dumps.
3. **CPU-Bound Offloading:** Any CPU-bound processing (AST chunking, embeddings inference) triggered by an MCP request *must* be offloaded to a blocking thread pool using `tokio::task::spawn_blocking`. The async worker handling the request will await the result of the blocking task, keeping the main event loop free to continue processing incoming frames from `stdin`.

## Consequences
- Eliminates the risk of partial reads or missed delimiters causing deadlocks.
- Guarantees the responsiveness of the MCP server, even under heavy computational load.
- Strict framing limits require that any massive chunks of context passed from the client must not exceed the maximum configured frame length.
