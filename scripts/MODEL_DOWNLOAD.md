# Stella ONNX Model Download Guide

This document describes how to download or copy the Stella embedding model for use with the Qdrant RAG MCP server.

## Quick Start

### Option 1: Copy from Local Cache (Recommended if available)

If you have pre-downloaded models in `/mnt/Meta/LLM/onnx` or similar location:

```bash
# Using the shell script
./scripts/download-stella.sh int8 --local /mnt/Meta/LLM/onnx

# Or directly with Python
python3 scripts/download_stella_model.py --quantization int8 --local-source /mnt/Meta/LLM/onnx
```

### Option 2: Download from Hugging Face

```bash
# Using the shell script
./scripts/download-stella.sh int8

# Or directly with Python
python3 scripts/download_stella_model.py --quantization int8
```

## Available Quantization Options

| Quantization | Size | Description |
|-------------|------|-------------|
| `int8` (default) | ~447 MB | INT8 quantized - best for CPU inference |
| `fp16` | ~875 MB | FP16 half precision |
| `q4` | ~391 MB | 4-bit quantized |
| `q4f16` | ~296 MB | 4-bit quantized with FP16 |
| `uint8` | ~447 MB | UINT8 quantized |
| `bnb4` | ~366 MB | BitsAndBytes 4-bit |
| `quantized` | ~447 MB | General quantized |

## Configuration

After downloading, ensure your `config/server_config.json` has the correct path:

```json
{
  "embeddings": {
    "model": "./data/models/stella_en_400M_v5/int8",
    "device": "cpu",
    "batch_size": 32,
    "normalize_embeddings": true,
    "query_prefix": "s2p_query: ",
    "requires_query_prefix": true
  }
}
```

## Model Files

The following files are downloaded/copied:

- `model.onnx` - The ONNX model file (required)
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings
- `config.json` - Model configuration
- `special_tokens_map.json` - Special tokens mapping
- `modules.json` - SentenceTransformers modules
- `config_sentence_transformers.json` - ST config
- `sentence_bert_config.json` - SentenceBERT config
- `vocab.txt` - Vocabulary file

## Requirements

- **hf CLI**: For downloading from Hugging Face
  - Install from: https://github.com/huggingface/hf-hub
  - Or use: `pip install huggingface_hub`

## Environment Variables

Optional environment variables:

- `HF_TOKEN` - Hugging Face token for rate limiting
- `QDRANT_EMBEDDING_DEVICE` - Device for embedding computation (default: cpu)

## Troubleshooting

### Model not found

Ensure the path in `config/server_config.json` matches the actual download location.

### Download fails

- Check your internet connection
- Verify HF_TOKEN is valid (if using authentication)
- Try reducing `--max-workers` for slower connections

### Out of memory

Use a smaller quantization (e.g., `q4f16`) or reduce batch size in config.
