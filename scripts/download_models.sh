#!/bin/bash

# scripts/download_models_simple.sh - Simplified version that works

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | grep -v '^$')
    set +a
fi

# Set cache directory - handle potential comments in env vars
TEMP_HOME="${SENTENCE_TRANSFORMERS_HOME:-~/Library/Caches/qdrant-mcp/models}"
# Remove any comments from the path
CACHE_DIR=$(echo "$TEMP_HOME" | sed 's/#.*//' | xargs)
CACHE_DIR="${CACHE_DIR/#\~/$HOME}"

echo -e "${BLUE}=== Embedding Model Downloader ===${NC}"
echo -e "${YELLOW}Cache directory: ${CACHE_DIR}${NC}"
echo ""

# Show current specialized embeddings configuration
echo -e "${GREEN}Current Specialized Embeddings Configuration:${NC}"
echo -e " Code: ${QDRANT_CODE_EMBEDDING_MODEL:-nomic-ai/CodeRankEmbed}"
echo -e " Config: ${QDRANT_CONFIG_EMBEDDING_MODEL:-jinaai/jina-embeddings-v3}"
echo -e " Documentation: ${QDRANT_DOC_EMBEDDING_MODEL:-hkunlp/instructor-large}"
echo -e " General: ${QDRANT_GENERAL_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
echo -e " Enabled: ${QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED:-true}"
echo ""

# Show ONNX models status
echo -e "${GREEN}ONNX Models Status:${NC}"
onnx_dir="${CACHE_DIR}/stella_en_400M_v5"
if [ -d "$onnx_dir" ]; then
echo -e " ${BLUE}Stella:${NC} Found at $onnx_dir"
if [ -f "${onnx_dir}/int8/model.onnx" ]; then
size=$(du -h "${onnx_dir}/int8/model.onnx" | cut -f1)
echo -e "  ${GREEN}✓ int8${NC} downloaded ($size)"
else
echo -e "  ${YELLOW}int8 not downloaded${NC}"
fi
else
echo -e " ${YELLOW}Stella ONNX models not downloaded${NC}"
fi
echo ""

# Specialized Embedding Models (Phase 2 - Recommended)
declare -a SPECIALIZED_MODELS=(
    "nomic-ai/CodeRankEmbed:2.0GB:🚀 Optimized for code understanding (PRIMARY for code)"
    "jinaai/jina-embeddings-v3:2.0GB:⚙️ Specialized for configuration files (PRIMARY for config)"
    "hkunlp/instructor-large:1.5GB:📚 Optimized for documentation with instruction support (PRIMARY for docs)"
    "sentence-transformers/all-MiniLM-L6-v2:90MB:🔗 General purpose and backward compatibility"
    "microsoft/codebert-base:440MB:💻 Fallback for code (if CodeRankEmbed fails)"
    "jinaai/jina-embeddings-v2-base-en:1.0GB:⚙️ Fallback for config (if v3 fails)"
    "sentence-transformers/all-mpnet-base-v2:420MB:📚 Fallback for documentation"
)

# General Purpose Models (Legacy - Single Model Mode)
declare -a GENERAL_MODELS=(
    "all-MiniLM-L6-v2:90MB:Fast, good for general use"
    "all-MiniLM-L12-v2:130MB:Better accuracy, still fast"
    "all-mpnet-base-v2:420MB:Excellent quality, general purpose"
    "all-distilroberta-v1:290MB:High quality, robust"
    "multi-qa-MiniLM-L6-cos-v1:90MB:Optimized for Q&A"
    "microsoft/codebert-base:440MB:Good for code understanding"
    "microsoft/unixcoder-base:550MB:Great for multiple programming languages"
    "Salesforce/codet5-small:220MB:Good for code generation"
    "intfloat/e5-large-v2:1.3GB:State-of-the-art quality"
    "BAAI/bge-large-en-v1.5:1.3GB:Excellent for retrieval"
)

# Function to check if model exists
check_model_exists() {
    local model=$1
    # Convert model name to directory format (org/model -> models--org--model)
    local model_dir="models--${model//\/--}"
    model_dir="${model_dir//\/$/}"  # Remove trailing slash if any
    
    # For sentence-transformers models, check with prefix
    if [[ "$model" == "sentence-transformers/"* ]]; then
        if [ -d "$CACHE_DIR/models--${model//\/--}" ]; then
            return 0
        fi
    fi
    
    # Check general pattern
    if [ -d "$CACHE_DIR/models--${model//\/--}" ]; then
        return 0
    fi
    
    return 1
}

# Function to get model size  
get_model_size() {
    local model=$1
    local model_dir="models--${model//\/--}"
    
    if [ -d "$CACHE_DIR/$model_dir" ]; then
        size=$(du -sh "$CACHE_DIR/$model_dir" | cut -f1)
        echo "$size"
        return
    fi
    
    echo "Not found"
}

# Display available models
echo -e "${GREEN}🚀 SPECIALIZED EMBEDDINGS (Phase 2 - Recommended):${NC}"
echo -e "${BLUE}Content-type specific models for optimal performance${NC}"
echo ""

total_models=0
declare -a ALL_MODELS=()

for i in "${!SPECIALIZED_MODELS[@]}"; do
    IFS=':' read -r model_name expected_size description <<< "${SPECIALIZED_MODELS[$i]}"
    ALL_MODELS+=("${SPECIALIZED_MODELS[$i]}")
    
    # Check if already downloaded
    if check_model_exists "$model_name"; then
        actual_size=$(get_model_size "$model_name")
        status="${GREEN}✓ Downloaded ($actual_size)${NC}"
    else
        status="${YELLOW}Not downloaded${NC}"
    fi
    
    printf "  %2d. %-35s %-8s %b\n" $((i+1)) "$model_name" "[$expected_size]" "$status"
    echo "      $description"
    echo ""
done

specialized_count=${#SPECIALIZED_MODELS[@]}
total_models=$specialized_count

echo -e "${GREEN}📚 GENERAL PURPOSE MODELS (Legacy/Single Model Mode):${NC}"
echo ""

for i in "${!GENERAL_MODELS[@]}"; do
    IFS=':' read -r model_name expected_size description <<< "${GENERAL_MODELS[$i]}"
    ALL_MODELS+=("${GENERAL_MODELS[$i]}")
    
    # Check if already downloaded
    if check_model_exists "$model_name"; then
        actual_size=$(get_model_size "$model_name")
        status="${GREEN}✓ Downloaded ($actual_size)${NC}"
    else
        status="${YELLOW}Not downloaded${NC}"
    fi
    
    printf "  %2d. %-35s %-8s %b\n" $((i+specialized_count+1)) "$model_name" "[$expected_size]" "$status"
    echo "      $description"
    echo ""
done

total_models=${#ALL_MODELS[@]}

echo -e "${BLUE}=== QUICK OPTIONS ===${NC}"
echo " 0. Download all specialized models (recommended)"
echo " 88. Download all models (specialized + general)"
echo " 99. Download custom model (enter name)"
echo ""
echo -e "${BLUE}=== ONNX MODEL DOWNLOADS ===${NC}"
echo " 100. Download Stella ONNX model (NovaSearch/stella_en_400M_v5)"
echo " 101. Download Llama Nemotron Rerank (cstr/llama-nemotron-rerank-1b-v2-ONNX)"
echo " 102. Download Jina Reranker v3 (keisuke-miyako/jina-reranker-v3-onnx-int8-NG)"
echo " 103. Download Sparse BM42 (Qdrant/all_miniLM_L6_v2_with_attentions)"
echo " 104. Download all ONNX models"
echo " 105. Download MiniLM-L12-v2 ONNX (keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16)"
echo " 106. Download CodeRankEmbed ONNX (mrsladoje/CodeRankEmbed-onnx-int8)"
echo ""

read -p "Select model(s) to download (comma-separated numbers): " choices

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check if required packages are installed
    python3 -c "import einops" 2>/dev/null || {
        echo -e "${YELLOW}Installing einops (required for some models)...${NC}"
        pip install einops >/dev/null 2>&1 || echo -e "${RED}Warning: Failed to install einops${NC}"
    }
    
    python3 -c "import InstructorEmbedding" 2>/dev/null || {
        echo -e "${YELLOW}Installing InstructorEmbedding (required for instructor models)...${NC}"
        pip install InstructorEmbedding >/dev/null 2>&1 || echo -e "${RED}Warning: Failed to install InstructorEmbedding${NC}"
    }
    
    echo ""
}

# Function to download a model
download_model() {
local model=$1

if check_model_exists "$model"; then
echo -e "${YELLOW}Model $model already exists. Skipping...${NC}"
return 181
fi

echo -e "${BLUE}Downloading $model...${NC}"

# Check if model needs trust_remote_code
local trust_remote_code="False"
if [[ "$model" == "nomic-ai/CodeRankEmbed" ]] || [[ "$model" == "jinaai/jina-embeddings-v3" ]]; then
trust_remote_code="True"
echo -e "${YELLOW}Note: This model requires trust_remote_code=True${NC}"
fi

# Use Python to download
python3 -c "
from sentence_transformers import SentenceTransformer
import sys
import os

model_name = '$model'
cache_dir = '$CACHE_DIR'
trust_remote = $trust_remote_code

try:
print(f'Downloading {model_name} from Hugging Face...')
if trust_remote:
model = SentenceTransformer(model_name, cache_folder=cache_dir, trust_remote_code=True)
else:
model = SentenceTransformer(model_name, cache_folder=cache_dir)
print(f'Successfully downloaded {model_name}')
except Exception as e:
print(f'Error: {e}')
sys.exit(1)
"

if [ $? -eq 0 ]; then
actual_size=$(get_model_size "$model")
echo -e "${GREEN}✓ $model downloaded successfully ($actual_size)${NC}"
else
echo -e "${RED}✗ Failed to download $model${NC}"
echo -e "${YELLOW}Tip: You may need to install additional dependencies or check your internet connection${NC}"
fi
}

# Function to download Stella ONNX model
download_stella_onnx() {
local quantization="${1:-int8}"
local output_dir="${CACHE_DIR}/stella_en_400M_v5/${quantization}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stella ONNX Model Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} NovaSearch/stella_en_400M_v5"
echo -e "${YELLOW}Quantization:${NC} ${quantization}"
echo -e "${YELLOW}

# Function to download Stella ONNX from Hugging Face
download_stella_onnx_from_hf() {
local quantization="$1"
local output_dir="$2"

# Check if hf CLI is available
if command -v hf &> /dev/null; then
echo "Using hf CLI to download..."

# Download ONNX model
local model_file="onnx/model_${quantization}.onnx"
if [ "$quantization" == "int8" ]; then
model_file="onnx/model_int8.onnx"
fi

hf download NovaSearch/stella_en_400M_v5 "$model_file" \
--revision 154316358e2c8bb71ef0fe7473fd12123acb293e \
--local-dir "$output_dir" 2>/dev/null

# Rename to model.onnx
if [ -f "${output_dir}/${model_file}" ]; then
mv "${output_dir}/${model_file}" "${output_dir}/model.onnx"
fi

# Download config files
hf download NovaSearch/stella_en_400M_v5 \
tokenizer.json tokenizer_config.json config.json \
special_tokens_map.json modules.json config_sentence_transformers.json \
--local-dir "$output_dir" 2>/dev/null

elif command -v huggingface-cli &> /dev/null; then
echo "Using huggingface-cli to download..."
huggingface-cli download NovaSearch/stella_en_400M_v5 \
--local-dir "$output_dir" \
--revision 154316358e2c8bb71ef0fe7473fd12123acb293e
else
echo -e "${RED}hf CLI not found. Please install from: https://github.com/huggingface/hf-hub${NC}"
echo "Or set STELLA_LOCAL_SOURCE to use a local copy."
return 1
fi
}

# Function to download Llama Nemotron Rerank ONNX
download_llama_nemotron_onnx() {
local quantization="${1:-int8}"
local output_dir="${CACHE_DIR}/llama-nemotron-rerank-1b-v2-ONNX/${quantization}"
local model_id="cstr/llama-nemotron-rerank-1b-v2-ONNX"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Llama Nemotron Rerank ONNX Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $model_id"
echo -e "${YELLOW}Quantization:${NC} ${quantization}"
echo -e "${YELLOW}Output:${NC} ${output_dir}"
echo ""

if [ -f "${output_dir}/model.onnx" ]; then
echo -e "${GREEN}✓ Already exists. Skipping...${NC}"
return 0
fi

# Check local source
LOCAL_SOURCE="${STELLA_LOCAL_SOURCE:-/mnt/Meta/LLM/onnx}"
if [ -d "${LOCAL_SOURCE}/llama-nemotron-rerank-1b-v2-ONNX" ]; then
echo "Copying from local cache..."
mkdir -p "$output_dir"
local source_dir="${LOCAL_SOURCE}/llama-nemotron-rerank-1b-v2-ONNX"
if [ -d "$source_dir" ]; then
cp -r "${source_dir}/"* "$output_dir/" 2>/dev/null || true
echo -e "${GREEN}✓ Copied from local source${NC}"
else
download_llama_nemotron_from_hf "$quantization" "$output_dir"
fi
else
download_llama_nemotron_from_hf "$quantization" "$output_dir"
fi

if [ -f "${output_dir}/model.onnx" ]; then
local size=$(du -h "${output_dir}/model.onnx" | cut -f1)
echo -e "${GREEN}✓ Downloaded successfully - ${size}${NC}"
else
echo -e "${RED}✗ Failed to download${NC}"
fi
}

download_llama_nemotron_from_hf() {
local quantization="$1"
local output_dir="$2"

if command -v hf &> /dev/null; then
hf download cstr/llama-nemotron-rerank-1b-v2-ONNX \
--local-dir "$output_dir" 2>/dev/null
elif command -v huggingface-cli &> /dev/null; then
huggingface-cli download cstr/llama-nemotron-rerank-1b-v2-ONNX \
--local-dir "$output_dir" 2>/dev/null
else
echo -e "${RED}hf CLI not found${NC}"
return 1
fi
}

# Function to download Jina Reranker v3 ONNX
download_jina_reranker_onnx() {
local output_dir="${CACHE_DIR}/jina-reranker-v3-onnx-int8-NG"
local model_id="keisuke-miyako/jina-reranker-v3-onnx-int8-NG"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Jina Reranker v3 ONNX Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $model_id"
echo -e "${YELLOW}Output:${NC} ${output_dir}"
echo ""

if [ -f "${output_dir}/model.onnx" ]; then
echo -e "${GREEN}✓ Already exists. Skipping...${NC}"
return 0
fi

LOCAL_SOURCE="${STELLA_LOCAL_SOURCE:-/mnt/Meta/LLM/onnx}"
if [ -d "${LOCAL_SOURCE}/jina-reranker-v3-onnx-int8-NG" ]; then
echo "Copying from local cache..."
mkdir -p "$output_dir"
cp -r "${LOCAL_SOURCE}/jina-reranker-v3-onnx-int8-NG/"* "$output_dir/" 2>/dev/null || true
echo -e "${GREEN}✓ Copied from local source${NC}"
else
if command -v hf &> /dev/null; then
hf download keisuke-miyako/jina-reranker-v3-onnx-int8-NG \
--local-dir "$output_dir" 2>/dev/null
elif command -v huggingface-cli &> /dev/null; then
huggingface-cli download keisuke-miyako/jina-reranker-v3-onnx-int8-NG \
--local-dir "$output_dir" 2>/dev/null
else
echo -e "${RED}hf CLI not found${NC}"
return 1
fi
fi

if [ -f "${output_dir}/model.onnx" ]; then
local size=$(du -h "${output_dir}/model.onnx" | cut -f1)
echo -e "${GREEN}✓ Downloaded successfully - ${size}${NC}"
else
echo -e "${RED}✗ Failed to download${NC}"
fi
}

# Function to download Sparse BM42
download_sparse_bm42() {
local output_dir="${CACHE_DIR}/qdrant_all_miniLM_L6_v2_with_attentions"
local model_id="Qdrant/all_miniLM_L6_v2_with_attentions"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sparse BM42 Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $model_id"
echo -e "${YELLOW}Output:${NC} ${output_dir}"
echo ""

if [ -d "${output_dir}/snapshots" ] || [ -f "${output_dir}/config.json" ]; then
echo -e "${GREEN}✓ Already exists. Skipping...${NC}"
return 0
fi

LOCAL_SOURCE="${STELLA_LOCAL_SOURCE:-/mnt/Meta/LLM/onnx}"
if [ -d "${LOCAL_SOURCE}/qdrant_all_miniLM_L6_v2_with_attentions" ]; then
echo "Copying from local cache..."
mkdir -p "$output_dir"
cp -r "${LOCAL_SOURCE}/qdrant_all_miniLM_L6_v2_with_attentions/"* "$output_dir/" 2>/dev/null || true
echo -e "${GREEN}✓ Copied from local source${NC}"
else
if command -v hf &> /dev/null; then
hf download Qdrant/all_miniLM_L6_v2_with_attentions \
--local-dir "$output_dir" 2>/dev/null
elif command -v huggingface-cli &> /dev/null; then
huggingface-cli download Qdrant/all_miniLM_L6_v2_with_attentions \
--local-dir "$output_dir" 2>/dev/null
else
echo -e "${RED}hf CLI not found${NC}"
return 1
fi
fi

echo -e "${GREEN}✓ Downloaded successfully${NC}"
}

# Function to download all ONNX models
download_all_onnx() {
echo -e "${BLUE}Downloading all ONNX models...${NC}"
download_stella_onnx "int8"
download_llama_nemotron_onnx "int8"
download_jina_reranker_onnx
download_sparse_bm42
download_minilm_l12_onnx
download_coderankembed_onnx
echo -e "${GREEN}✓ All ONNX models processed${NC}"
}

# Function to download MiniLM-L12-v2 ONNX
download_minilm_l12_onnx() {
local output_dir="${CACHE_DIR}/all-MiniLM-L12-v2-onnx-fp16"
local model_id="keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MiniLM-L12-v2 ONNX Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $model_id"
echo -e "${YELLOW}Output:${NC} ${output_dir}"
echo ""

if [ -f "${output_dir}/model.onnx" ] || [ -d "${output_dir}/snapshots" ]; then
echo -e "${GREEN}✓ Already exists. Skipping...${NC}"
return 0
fi

LOCAL_SOURCE="${STELLA_LOCAL_SOURCE:-/mnt/Meta/LLM/onnx}"
if [ -d "${LOCAL_SOURCE}/all-MiniLM-L12-v2-onnx-fp16" ]; then
echo "Copying from local cache..."
mkdir -p "$output_dir"
cp -r "${LOCAL_SOURCE}/all-MiniLM-L12-v2-onnx-fp16/"* "$output_dir/" 2>/dev/null || true
echo -e "${GREEN}✓ Copied from local source${NC}"
else
if command -v hf &> /dev/null; then
hf download keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16 \
--local-dir "$output_dir" 2>/dev/null
elif command -v huggingface-cli &> /dev/null; then
huggingface-cli download keisuke-miyako/all-MiniLM-L12-v2-onnx-fp16 \
--local-dir "$output_dir" 2>/dev/null
else
echo -e "${RED}hf CLI not found${NC}"
return 1
fi
fi

if [ -f "${output_dir}/model.onnx" ] || [ -d "${output_dir}/snapshots" ]; then
echo -e "${GREEN}✓ Downloaded successfully${NC}"
else
echo -e "${RED}✗ Failed to download${NC}"
fi
}

# Function to download CodeRankEmbed ONNX
download_coderankembed_onnx() {
local output_dir="${CACHE_DIR}/CodeRankEmbed-onnx-int8"
local model_id="mrsladoje/CodeRankEmbed-onnx-int8"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CodeRankEmbed ONNX Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $model_id"
echo -e "${YELLOW}Purpose:${NC} Code embeddings (ONNX INT8)"
echo -e "${YELLOW}Output:${NC} ${output_dir}"
echo ""

if [ -f "${output_dir}/model.onnx" ] || [ -d "${output_dir}/snapshots" ]; then
echo -e "${GREEN}✓ Already exists. Skipping...${NC}"
return 0
fi

LOCAL_SOURCE="${STELLA_LOCAL_SOURCE:-/mnt/Meta/LLM/onnx}"
if [ -d "${LOCAL_SOURCE}/CodeRankEmbed-onnx-int8" ]; then
echo "Copying from local cache..."
mkdir -p "$output_dir"
cp -r "${LOCAL_SOURCE}/CodeRankEmbed-onnx-int8/"* "$output_dir/" 2>/dev/null || true
echo -e "${GREEN}✓ Copied from local source${NC}"
else
if command -v hf &> /dev/null; then
hf download mrsladoje/CodeRankEmbed-onnx-int8 \
--local-dir "$output_dir" 2>/dev/null
elif command -v huggingface-cli &> /dev/null; then
huggingface-cli download mrsladoje/CodeRankEmbed-onnx-int8 \
--local-dir "$output_dir" 2>/dev/null
else
echo -e "${RED}hf CLI not found${NC}"
return 1
fi
fi

if [ -f "${output_dir}/model.onnx" ] || [ -d "${output_dir}/snapshots" ]; then
echo -e "${GREEN}✓ Downloaded successfully${NC}"
else
echo -e "${RED}✗ Failed to download${NC}"
fi
}

# Check dependencies first
check_dependencies

# Process user choices
IFS=',' read -ra SELECTED <<< "$choices"

for choice in "${SELECTED[@]}"; do
choice=$(echo $choice | xargs) # Trim whitespace

if [ "$choice" = "0" ]; then
# Download all specialized models (recommended)
echo -e "${BLUE}Downloading all specialized models - recommended...${NC}"
for i in "${!SPECIALIZED_MODELS[@]}"; do
IFS=':' read -r model_name _ _ <<< "${SPECIALIZED_MODELS[$i]}"
download_model "$model_name"
echo ""
done
elif [ "$choice" = "88" ]; then
# Download all models (specialized + general)
echo -e "${BLUE}Downloading all models (specialized + general)...${NC}"
for i in "${!ALL_MODELS[@]}"; do
IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$i]}"
download_model "$model_name"
echo ""
done
elif [ "$choice" = "99" ]; then
# Custom model
echo ""
read -p "Enter model name: " custom_model
if [ -n "$custom_model" ]; then
download_model "$custom_model"
fi
elif [ "$choice" = "100" ]; then
# Download Stella ONNX model
echo ""
read -p "Enter quantization (int8|fp16|q4|q4f16|uint8|bnb4): " stella_quant
stella_quant="${stella_quant:-int8}"
download_stella_onnx "$stella_quant"
elif [ "$choice" = "101" ]; then
# Download Llama Nemotron Rerank
echo ""
read -p "Enter quantization (int8): " llama_quant
llama_quant="${llama_quant:-int8}"
download_llama_nemotron_onnx "$llama_quant"
elif [ "$choice" = "102" ]; then
# Download Jina Reranker v3
download_jina_reranker_onnx
elif [ "$choice" = "103" ]; then
# Download Sparse BM42
download_sparse_bm42
elif [ "$choice" = "104" ]; then
# Download all ONNX models
download_all_onnx
elif [ "$choice" = "105" ]; then
# Download MiniLM-L12-v2 ONNX
download_minilm_l12_onnx
elif [ "$choice" = "106" ]; then
# Download CodeRankEmbed ONNX
download_coderankembed_onnx
elif [ "$choice" -ge 1 ] && [ "$choice" -le "$total_models" ]; then
# Download selected
IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$((choice-1))]}"
download_model "$model_name"
echo ""
fi
done
    elif [ "$choice" = "88" ]; then
        # Download all models (specialized + general)
        echo -e "${BLUE}Downloading all models (specialized + general)...${NC}"
        for i in "${!ALL_MODELS[@]}"; do
            IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$i]}"
            download_model "$model_name"
            echo ""
        done
    elif [ "$choice" = "99" ]; then
        # Custom model
        echo ""
        read -p "Enter model name: " custom_model
        if [ -n "$custom_model" ]; then
            download_model "$custom_model"
        fi
    elif [ "$choice" -ge 1 ] && [ "$choice" -le "$total_models" ]; then
        # Download selected
        IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$((choice-1))]}"
        download_model "$model_name"
        echo ""
    fi
done

# Show final status
echo ""
echo -e "${GREEN}=== Download Summary ===${NC}"
echo -e "${YELLOW}Models stored in: ${CACHE_DIR}${NC}"
echo ""

# Show ONNX models status
echo -e "${GREEN}ONNX Models Status:${NC}"
onnx_base_dir="${CACHE_DIR}"

# Stella
stella_dir="${onnx_base_dir}/stella_en_400M_v5"
if [ -d "$stella_dir" ]; then
echo -e " ${BLUE}Stella:${NC} Found"
for quant_dir in "$stella_dir"/*/; do
if [ -d "$quant_dir" ] && [ -f "${quant_dir}model.onnx" ]; then
size=$(du -h "${quant_dir}model.onnx" | cut -f1)
echo -e "   ${GREEN}✓ $(basename $quant_dir)${NC} ($size)"
fi
done
fi

# Llama Nemotron
llama_dir="${onnx_base_dir}/llama-nemotron-rerank-1b-v2-ONNX"
if [ -d "$llama_dir" ]; then
echo -e " ${BLUE}Llama Nemotron Rerank:${NC} Found"
for quant_dir in "$llama_dir"/*/; do
if [ -d "$quant_dir" ] && [ -f "${quant_dir}model.onnx" ]; then
size=$(du -h "${quant_dir}model.onnx" | cut -f1)
echo -e "   ${GREEN}✓ $(basename $quant_dir)${NC} ($size)"
fi
done
fi

# Jina Reranker
jina_dir="${onnx_base_dir}/jina-reranker-v3-onnx-int8-NG"
if [ -d "$jina_dir" ] && [ -f "$jina_dir/model.onnx" ]; then
size=$(du -h "$jina_dir/model.onnx" | cut -f1)
echo -e " ${BLUE}Jina Reranker v3:${NC} ${GREEN}✓${NC} ($size)"
fi

# Sparse BM42
sparse_dir="${onnx_base_dir}/qdrant_all_miniLM_L6_v2_with_attentions"
if [ -d "$sparse_dir" ]; then
echo -e " ${BLUE}Sparse BM42:${NC} ${GREEN}✓${NC}"
fi

echo ""

# List all downloaded models
echo "Downloaded models:"
found_models=false

# Create a temporary file to store model info
temp_file=$(mktemp)

# Find all models
for dir in "$CACHE_DIR"/models--*; do
    if [ -d "$dir/snapshots" ]; then
        dir_name=$(basename "$dir")
        size=$(du -sh "$dir" | cut -f1)
        
        # Extract model name
        model_name=""
        case "$dir_name" in
            models--sentence-transformers--*)
                model_name="${dir_name#models--sentence-transformers--}"
                ;;
            models--microsoft--*)
                model_name="microsoft/${dir_name#models--microsoft--}"
                ;;
            models--Salesforce--*)
                model_name="Salesforce/${dir_name#models--Salesforce--}"
                ;;
            models--intfloat--*)
                model_name="intfloat/${dir_name#models--intfloat--}"
                ;;
            models--BAAI--*)
                model_name="BAAI/${dir_name#models--BAAI--}"
                ;;
            models--nomic-ai--*)
                model_name="nomic-ai/${dir_name#models--nomic-ai--}"
                ;;
            models--jinaai--*)
                model_name="jinaai/${dir_name#models--jinaai--}"
                ;;
            models--hkunlp--*)
                model_name="hkunlp/${dir_name#models--hkunlp--}"
                ;;
        esac
        
        if [ -n "$model_name" ]; then
            echo "  - $model_name ($size)"
            echo "$model_name" >> "$temp_file"
            found_models=true
        fi
    fi
done

if [ "$found_models" = false ]; then
    echo "  (none)"
fi

# Show total disk usage
echo ""
if [ -d "$CACHE_DIR" ]; then
    total_size=$(du -sh "$CACHE_DIR" | cut -f1)
    echo -e "${BLUE}Total disk usage: ${total_size}${NC}"
fi

# Create model list file
echo ""
echo -e "${YELLOW}Creating model list...${NC}"
cp "$temp_file" "$CACHE_DIR/downloaded_models.txt"
echo -e "${GREEN}Model list saved to: $CACHE_DIR/downloaded_models.txt${NC}"

# Offer to update .env
echo ""
read -p "Would you like to set one of these models as default in .env? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Select a model to set as default:"
    
    # Read models from temp file
    i=1
    declare -a available_models=()
    while IFS= read -r model; do
        echo "  $i. $model"
        available_models+=("$model")
        ((i++))
    done < "$temp_file"
    
    if [ ${#available_models[@]} -gt 0 ]; then
        read -p "Enter number: " model_choice
        
        if [ "$model_choice" -ge 1 ] && [ "$model_choice" -le "${#available_models[@]}" ]; then
            selected_model="${available_models[$((model_choice-1))]}"
            
            # Update .env
            if grep -q "EMBEDDING_MODEL=" .env 2>/dev/null; then
                sed -i.bak "s|EMBEDDING_MODEL=.*|EMBEDDING_MODEL=$selected_model|" .env
                echo -e "${GREEN}Updated EMBEDDING_MODEL in .env to: $selected_model${NC}"
            else
                echo "EMBEDDING_MODEL=$selected_model" >> .env
                echo -e "${GREEN}Added EMBEDDING_MODEL to .env: $selected_model${NC}"
            fi
        fi
    fi
fi

# Clean up
rm -f "$temp_file"

echo ""
echo -e "${GREEN}Done!${NC}"
