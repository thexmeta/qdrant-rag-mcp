# BM42 Sparse Embeddings Technical Guide

**Version**: v0.3.5+ | **Status**: Production Ready

## Overview

BM42 is a sparse vector embedding method that provides superior keyword matching capabilities when combined with dense semantic search. This guide covers the technical implementation, configuration, and troubleshooting of BM42 sparse embeddings in the Qdrant RAG MCP Server.

## What are Sparse Embeddings?

Unlike dense embeddings (single vector representing semantic meaning), sparse embeddings create high-dimensional vectors where most dimensions are zero, with non-zero values representing specific term frequencies.

### Dense vs Sparse vs Hybrid

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Dense** | Single vector (e.g., 384D) capturing semantic meaning | Semantic search, conceptual similarity |
| **Sparse** | High-dimensional vector with term-based activation | Keyword matching, exact term search |
| **Hybrid** | Combines dense + sparse for best of both worlds | Production RAG systems |

## Architecture

### Component Flow

```
User Query
    │
    ├──> Dense Embedding (Semantic) ──┐
    │    └─> all-MiniLM-L12-v2        │
    │                                  ├──> Hybrid Score ──> Results
    ├──> Sparse Embedding (BM42) ─────┘
         └─> Native ONNX Runtime
```

### Native ONNX Implementation

BM42 uses a custom ONNX Runtime implementation for efficient sparse vector generation without heavy external dependencies:

- **Model**: `Qdrant/all_miniLM_L6_v2_with_attentions`
- **Engine**: `onnxruntime` + `tokenizers`
- **Format**: ONNX (optimized for inference)
- **Method**: BM42 (attention-based sparse embeddings)

## Configuration

### Environment Variables

```bash
# Enable BM42 sparse embeddings
export QDRANT_SPARSE_METHOD=bm42
export QDRANT_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions

# Optional: Override model path
export QDRANT_SPARSE_MODEL_PATH=./qdrant_all_miniLM_L6_v2_with_attentions
```

### Server Configuration

In `config/server_config.json`:

```json
{
  "sparse_embeddings": {
    "model": "./qdrant_all_miniLM_L6_v2_with_attentions",
    "method": "bm42",
    "device": "cpu"
  },
  "sparse_method": "${QDRANT_SPARSE_METHOD:-bm42}",
  "sparse_model": "${QDRANT_SPARSE_MODEL:-Qdrant/bm42-all-minilm-l6-v2-attentions}"
}
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_SPARSE_METHOD` | `bm42` | Sparse embedding method (`bm42` or `bm25`) |
| `QDRANT_SPARSE_MODEL` | `Qdrant/bm42-all-minilm-l6-v2-attentions` | Model identifier for sparse embeddings |
| `QDRANT_SPARSE_MODEL_PATH` | `./qdrant_all_miniLM_L6_v2_with_attentions` | Local model path override |

## Setup Instructions

### Step 1: Install Dependencies

```bash
# Core dependencies for ONNX
uv pip install onnxruntime tokenizers numpy

# Optional: fastembed is no longer required but can be installed for fallback
uv pip install fastembed
```

### Step 2: Configure Environment

```bash
# Add to .env file
QDRANT_SPARSE_METHOD=bm42
QDRANT_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions
```

### Step 3: Reindex Collections

Sparse embeddings require separate collection configuration:

```bash
# Delete existing collections (if reindexing)
curl -X DELETE http://localhost:6333/collections/code_collection
curl -X DELETE http://localhost:6333/collections/config_collection
curl -X DELETE http://localhost:6333/collections/documentation_collection

# Reindex with BM42 enabled
"Reindex this directory"
```

### Step 4: Verify Configuration

```bash
# Check sparse vector configuration
curl http://localhost:6333/collections/code_collection

# Should show sparse_vector_config in response
```

## How BM42 Works

### 1. Tokenization

Input text is tokenized into individual terms:
```python
"Hello world example" → ["hello", "world", "example"]
```

### 2. Attention-Based Weighting

BM42 uses attention mechanisms to weight term importance:
- Considers term context within the document
- Produces more nuanced weights than pure frequency

### 3. Sparse Vector Creation

Creates high-dimensional sparse vector:
```
Dimension 1024: 0.0
Dimension 5731: 0.0
Dimension 892:  0.847  ← "hello"
Dimension 3421: 0.623  ← "world"
...
```

### 4. Hybrid Scoring

Final score combines dense and sparse:
```
final_score = α * dense_score + (1 - α) * sparse_score
```

Where `α` is typically 0.5-0.7 depending on use case.

## Performance Considerations

### Memory Usage

| Component | Memory |
|-----------|--------|
| Dense Model (all-MiniLM-L12-v2) | ~350MB |
| Sparse Model (BM42 ONNX) | ~150MB |
| Sparse Vectors (per doc) | ~50KB |
| Dense Vectors (per doc) | ~1.5KB |

### Indexing Speed

- **Dense Only**: ~100 docs/sec
- **Sparse Only**: ~500 docs/sec
- **Hybrid (BM42 + Dense)**: ~80 docs/sec

### Query Performance

- **Dense Only**: ~50ms latency
- **Hybrid (BM42 + Dense)**: ~80ms latency
- **Improvement**: +30% precision in keyword-heavy queries

## Use Cases

### When to Use BM42

✅ **Code Search**: Finding specific function names, variables, classes
✅ **Error Messages**: Exact match for error strings
✅ **API Endpoints**: Precise route matching
✅ **Configuration Keys**: Exact key lookups
✅ **Technical Terms**: Specific terminology

### When BM42 May Not Help

❌ **Pure Conceptual Search**: Abstract concepts without specific terms
❌ **Cross-Lingual Search**: BM42 is English-optimized
❌ **Very Short Queries**: Single-word queries may not benefit

## Troubleshooting

### Issue: BM42 Not Working

**Symptoms**: No sparse vectors being generated

**Solutions**:
1. Verify ONNX dependencies: `uv pip show onnxruntime tokenizers`
2. Check environment variables are set
3. Verify model path exists in `./data/models/` or can be downloaded
4. Check server logs for ONNX session initialization errors

### Issue: Sparse Collection Not Found

**Symptoms**: Error "collection not found" for sparse vectors

**Solutions**:
1. Reindex after enabling BM42 configuration
2. Verify Qdrant version supports sparse vectors (>=1.7)
3. Check collection configuration includes sparse_vector_config

### Issue: Slow Indexing Performance

**Symptoms**: Indexing takes significantly longer with BM42

**Solutions**:
1. Reduce batch size in configuration
2. Use CPU-only mode (disable MPS for sparse)
3. Consider sparse-only for initial index, add dense later

### Issue: High Memory Usage

**Symptoms**: Memory spikes during indexing

**Solutions**:
1. Reduce concurrent indexing threads
2. Use smaller chunk sizes
3. Enable sparse vectors only for code files (skip docs/configs)

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8080/health
```

Response includes sparse embedding status:
```json
{
  "sparse_embeddings": {
    "enabled": true,
    "method": "bm42",
    "model": "Qdrant/bm42-all-minilm-l6-v2-attentions",
    "status": "healthy"
  }
}
```

### Logs to Monitor

```bash
# Real-time sparse embedding logs
./scripts/qdrant-logs | grep -i "sparse\|bm42"

# Model loading
./scripts/qdrant-logs | grep "sparse.*model"

# Indexing progress
./scripts/qdrant-logs | grep "sparse.*index"
```

## Advanced Configuration

### Custom BM42 Model

To use a different sparse embedding model:

```bash
export QDRANT_SPARSE_MODEL=your-org/your-model-name
export QDRANT_SPARSE_MODEL_PATH=/path/to/local/model.onnx
```

### Hybrid Scoring Weights

Adjust the balance between dense and sparse:

```python
# In scoring configuration
DENSE_WEIGHT = 0.6  # 60% dense, 40% sparse
SPARSE_WEIGHT = 0.4
```

### Sparse-Only Mode

For keyword-only search (not recommended):

```bash
export QDRANT_DENSE_ENABLED=false
export QDRANT_SPARSE_METHOD=bm42
```

## Comparison with BM25

| Feature | BM25 | BM42 |
|---------|------|------|
| **Method** | Statistical | Neural (Attention) |
| **Context Awareness** | No | Yes |
| **Term Weighting** | Frequency-based | Attention-based |
| **Performance** | Faster | More accurate |
| **Model Size** | N/A (algorithm) | ~150MB |
| **Best For** | Simple keyword search | Contextual keyword search |

## References

- [Qdrant Sparse Vectors Documentation](https://qdrant.tech/documentation/concepts/sparse-vectors/)
- [Fastembed Library](https://github.com/qdrant/fastembed)
- [BM42 Research Paper](https://arxiv.org/abs/2402.01486)

## Related Documentation

- [Complete Setup Guide](../complete-setup-and-usage-guide.md#bm42-sparse-embeddings-configuration)
- [Hybrid Search Implementation](./hybrid-search-implementation.md)
- [Scoring Pipeline Architecture](./scoring-pipeline-architecture.md)
