# ADR 0001: Architecture Migration Strategy

## Status
Accepted

## Context
We are migrating a ~24k LOC Python application (Qdrant RAG MCP Server) to Rust. The Python application involves complex components such as specialized AST chunking (for multiple programming languages), hybrid search using Qdrant and BM25, and direct interaction with the MCP (Model Context Protocol) via `stdio`.

A monolithic rewrite is unfeasible due to the size of the codebase and the significant ecosystem impedance mismatch between Python (dynamically typed, heavy reliance on PyTorch/HuggingFace libraries) and Rust (statically typed, borrow checker, different ML ecosystem).

## Decision
We will employ a multi-phase, abstraction-first migration strategy utilizing isolated, verifiable modules inside a parallel directory structure (`/rust-rewrite`). This allows for side-by-side integration testing against the Python baseline.

### Migration Phases
1. **Phase 1 (Core):** Implement the `stdio` MCP interface, JSON-RPC routing, and the Qdrant gRPC client integration.
2. **Phase 2 (Processing):** Implement multi-language AST parsing using `tree-sitter` and core text chunking logic.
3. **Phase 3 (ML/Search):** Integrate local embeddings inference and hybrid search algorithms.
4. **Phase 4 (Integration):** Implement the Github integration module and progressive context logic.

### Ecosystem Equivalents
We have chosen the following Rust equivalents for the core Python dependencies:
- **Embeddings (`sentence-transformers`/`torch`):** `ort` (ONNX Runtime). This avoids the heavy build overhead of PyTorch bindings in Rust (`tch-rs`). Existing models will need to be exported to ONNX format.
- **AST Parsing (`ast`):** `tree-sitter-rs`. Mandatory for robust, language-agnostic AST parsing (JS, Python, Rust, Go).
- **Search Engine (`rank-bm25`):** `tantivy`. A robust search engine to replicate BM25 functionality. (Note: rigorous benchmarking against the Python BM25 logic is required to prevent algorithmic drift).
- **Text Splitting (`langchain-text-splitters`):** `text-splitter` crate. Native recursive character splitting.
- **Qdrant (`qdrant-client`):** The official `qdrant-client` Rust crate (gRPC-based).

## Consequences
- Requires upfront validation of embedding models to ensure they can be successfully exported to ONNX. If they cannot, we must pivot to an external inference server (e.g., TEI or `llama.cpp`).
- Necessitates rigorous A/B benchmarking for BM25 search results to guarantee functional parity.
- Provides a significantly faster, memory-safe, and highly concurrent execution environment compared to the original Python implementation.
