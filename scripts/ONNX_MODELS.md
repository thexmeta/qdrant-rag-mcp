# ONNX Model Management Guide

## Overview

This project now supports multiple ONNX models for various tasks:

| Model | Repository | Purpose | Size |
|-------|------------|---------|------|
| **Stella** | `NovaSearch/stella_en_400M_v5` | General embeddings | 426MB (int8) |
| **Llama Nemotron Rerank** | `cstr/llama-nemotron-rerank-1b-v2-ONNX` | Reranking | 1.2GB |
| **Jina Reranker v3** | `keisuke-miyako/jina-reranker-v3-onnx-int8-NG` | Reranking | 574MB |
| **Sparse BM42** | `Qdrant/all_miniLM_L6_v2_with_attentions` | Sparse embeddings | ~100MB |
| **MiniLM-L12-v2** | `keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16` | General embeddings (FP16) | ~260MB |
| **CodeRankEmbed** | `mrsladoje/CodeRankEmbed-onnx-int8` | Code embeddings (INT8) | ~300MB |

## Quick Start

### Download All ONNX Models

```bash
cd /mnt/Meta/Projects/Python/qdrant-rag-mcp
./scripts/download_models.sh
# Select option 104
```

### Download Individual Models

```bash
# Stella (with quantization options)
./scripts/download_models.sh
# Select option 100, then choose quantization (int8/fp16/q4/etc.)

# Llama Nemotron Rerank
./scripts/download_models.sh
# Select option 101

# Jina Reranker v3
./scripts/download_models.sh
# Select option 102

# Sparse BM42
./scripts/download_models.sh
# Select option 103

# MiniLM-L12-v2 ONNX
./scripts/download_models.sh
# Select option 105

# CodeRankEmbed ONNX
./scripts/download_models.sh
# Select option 106
```

### List Downloaded Models

```bash
./scripts/manage_models.sh list
```

## Model Locations

Models are stored in:
```
/mnt/Meta/Projects/Python/qdrant-rag-mcp/data/models/
├── stella_en_400M_v5/
│   ├── int8/
│   │   └── model.onnx
│   ├── fp16/
│   │   └── model.onnx
│   └── ...
├── llama-nemotron-rerank-1b-v2-ONNX/
│   └── int8/
│       └── model.onnx
├── jina-reranker-v3-onnx-int8-NG/
│   └── model.onnx
└── qdrant_all_miniLM_L6_v2_with_attentions/
    └── ...
```

## Configuration

Update `config/server_config.json` to use the downloaded models:

```json
{
  "embeddings": {
    "model": "./data/models/stella_en_400M_v5/int8"
  },
  "search": {
    "reranker_model": "./data/models/llama-nemotron-rerank-1b-v2-ONNX/int8"
  },
  "hybrid_search": {
    "sparse_model": "./data/models/qdrant_all_miniLM_L6_v2_with_attentions"
  }
}
```

## Stella Quantization Options

| Quantization | Size | Use Case |
|-------------|------|----------|
| `int8` | 426MB | Best for CPU inference (recommended) |
| `fp16` | 835MB | GPU with FP16 support |
| `q4` | 391MB | 4-bit quantized |
| `q4f16` | 296MB | 4-bit with FP16 |
| `uint8` | 426MB | Alternative INT8 |
| `bnb4` | 366MB | BitsAndBytes 4-bit |

## Local Cache

If you have models in `/mnt/Meta/LLM/onnx/`, the script will automatically copy from there instead of downloading.

Set environment variable to use custom local source:
```bash
export STELLA_LOCAL_SOURCE=/path/to/local/models
```

## Requirements

- **hf CLI**: For downloading from Hugging Face
  - Install: `pip install huggingface_hub` or from https://github.com/huggingface/hf-hub

## Troubleshooting

### Models not detected
Ensure models are in the correct directory:
```bash
ls -la data/models/
```

### Download fails
- Check internet connection
- Verify hf CLI is installed: `hf --version`
- Try using local cache if available

### Out of memory
Use smaller quantization (e.g., `int8` or `q4f16`) or reduce batch size in config.
