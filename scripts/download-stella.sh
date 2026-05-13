#!/bin/bash
# Download Stella ONNX model from Hugging Face or local cache
# Usage: ./download-stella.sh [int8|fp16|q4|q4f16|uint8|bnb4|quantized] [--local /path/to/models]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default quantization
QUANTIZATION="${1:-int8}"

# Check for --local flag
LOCAL_SOURCE=""
if [[ "$2" == "--local" ]]; then
    LOCAL_SOURCE="${3:-}"
    if [ -z "$LOCAL_SOURCE" ]; then
        # Try common locations
        if [ -d "/mnt/Meta/LLM/onnx" ]; then
            LOCAL_SOURCE="/mnt/Meta/LLM/onnx"
        elif [ -d "$HOME/.cache/huggingface/hub" ]; then
            LOCAL_SOURCE="$HOME/.cache/huggingface/hub"
        fi
    fi
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stella ONNX Model Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Quantization:${NC} ${QUANTIZATION}"

if [ -n "$LOCAL_SOURCE" ]; then
    echo -e "${YELLOW}Local Source:${NC} ${LOCAL_SOURCE}"
else
    echo -e "${YELLOW}Output:${NC} ${PROJECT_ROOT}/data/models/stella_en_400M_v5/${QUANTIZATION}"
fi

echo ""

# Build command
CMD="python3 ${SCRIPT_DIR}/download_stella_model.py --quantization ${QUANTIZATION}"

if [ -n "$LOCAL_SOURCE" ]; then
    CMD="${CMD} --local-source ${LOCAL_SOURCE}"
fi

# Run the Python script
eval "$CMD"
