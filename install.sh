#!/bin/bash
# Unified installation script for Qdrant RAG MCP Server
# Supports Linux and macOS. Uses 'uv' for robust environment management.

set -e

# Use /tmp for cache and /mnt/Meta for tools to avoid root partition space issues
export UV_CACHE_DIR="/tmp/uv_cache"
export UV_TOOL_DIR="/mnt/Meta/.uv_tools"
export UV_TOOL_BIN_DIR="$HOME/.local/bin" # Keep bin in home, but actual tool data in /mnt/Meta
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$UV_TOOL_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Qdrant RAG MCP Server - Installation${NC}"
echo "=========================================="

# 1. Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env 2>/dev/null || true
    # Add to path for current session if just installed
    export PATH="$HOME/.local/bin:$PATH"
fi

echo -e "${GREEN}✅ uv is ready: $(uv --version)${NC}"

# 2. Prepare environment
echo -e "${BLUE}📦 Preparing environment and dependencies...${NC}"
uv sync --all-extras

# 3. Install as a global tool
echo -e "${BLUE}🔧 Installing qdrant-rag-mcp as a global tool...${NC}"
# This creates a 'qdrant-rag-mcp' command in ~/.local/bin
uv tool install --force --editable .

# 4. Verify the command exists
if ! command -v qdrant-rag-mcp &> /dev/null; then
    # Try adding ~/.local/bin to PATH if not there
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v qdrant-rag-mcp &> /dev/null; then
        echo -e "${RED}❌ Failed to register 'qdrant-rag-mcp' command.${NC}"
        echo "Please ensure ~/.local/bin is in your PATH."
        exit 1
    fi
fi

COMMAND_PATH=$(command -v qdrant-rag-mcp)
echo -e "${GREEN}✅ Command registered at: $COMMAND_PATH${NC}"

# 5. Handle the legacy /usr/bin/qdrant-rag-mcp if requested
if [ "$1" == "--fix-usr-bin" ]; then
    echo -e "${BLUE}🔧 Creating symlink in /usr/local/bin for legacy support...${NC}"
    sudo ln -sf "$COMMAND_PATH" /usr/local/bin/qdrant-rag-mcp
    echo -e "${GREEN}✅ Symlink created at /usr/local/bin/qdrant-rag-mcp${NC}"
fi

echo ""
echo -e "${GREEN}✨ Installation Complete!${NC}"
echo "------------------------------------------"
echo "You can now use the 'qdrant-rag-mcp' command from anywhere."
echo "If you saw an error about /usr/bin/qdrant-rag-mcp, you should now update"
echo "your Claude configuration to use the new path: $COMMAND_PATH"
echo ""
echo "To test:"
echo "  qdrant-rag-mcp --help"
echo ""
