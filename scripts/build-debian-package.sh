#!/bin/bash
set -e

# Configuration
PKG_NAME="qdrant-rag-mcp"
VERSION=$(grep -m 1 "version =" pyproject.toml | cut -d '"' -f 2)
ARCH="amd64"
BUILD_ROOT="build/deb"
PKG_DIR="${BUILD_ROOT}/${PKG_NAME}_${VERSION}_${ARCH}"
RELEASE_DIR="releases"

echo "🚀 Building Debian package for ${PKG_NAME} v${VERSION}..."

# Clean up previous builds
rm -rf "$BUILD_ROOT"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/opt/${PKG_NAME}"
mkdir -p "$RELEASE_DIR"

# Copy source files and essential directories
echo "📦 Copying files to package directory..."
cp -r src "$PKG_DIR/opt/${PKG_NAME}/"
cp -r scripts "$PKG_DIR/opt/${PKG_NAME}/"
[ -d config ] && cp -r config "$PKG_DIR/opt/${PKG_NAME}/"
cp pyproject.toml uv.lock README.md "$PKG_DIR/opt/${PKG_NAME}/"

# Create a wrapper script in /usr/bin
echo "🔧 Creating wrapper script..."
cat > "$PKG_DIR/usr/bin/${PKG_NAME}" << 'EOF'
#!/bin/bash
# Wrapper for qdrant-rag-mcp
# This script ensures the server runs from /opt/qdrant-rag-mcp using the global uv environment

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' not found. This package requires 'uv' (ultraviolet) to manage its environment." >&2
    echo "   Please install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# IMPORTANT: Save current directory for context detection
export MCP_CLIENT_CWD="$(pwd)"

# Load configuration from common locations
CONFIG_LOCATIONS=(
    "$HOME/.qdrant-rag-mcp/.env"
    "$HOME/.config/qdrant-rag-mcp/.env"
    "/opt/qdrant-rag-mcp/.env"
    "$(pwd)/.env"
)

for LOC in "${CONFIG_LOCATIONS[@]}"; do
    if [ -f "$LOC" ]; then
        # echo "ℹ️  Loading config from $LOC" >&2
        set -a
        source "$LOC"
        set +a
        break
    fi
done

# Run the server directly using the venv binary if it exists, otherwise fall back to uv run
if [ -f "/opt/qdrant-rag-mcp/.venv/bin/qdrant-rag-mcp" ]; then
    exec /opt/qdrant-rag-mcp/.venv/bin/qdrant-rag-mcp "$@"
else
    exec uv run --directory "/opt/qdrant-rag-mcp" qdrant-rag-mcp "$@"
fi
EOF
chmod +x "$PKG_DIR/usr/bin/${PKG_NAME}"

# Create the control file
echo "📝 Generating DEBIAN/control file..."
cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Qdrant RAG Team
Depends: curl, ca-certificates
Description: Qdrant RAG MCP Server
 A context-aware Model Context Protocol (MCP) server that provides semantic
 search capabilities across your codebase using Qdrant vector database.
 Optimized for use with Claude Code and other MCP clients.
EOF

# Create a postinst script
echo "📜 Generating DEBIAN/postinst script..."
cat > "$PKG_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e
echo "🔧 Initializing environment for qdrant-rag-mcp..."
echo "📦 This may take a few minutes as it downloads large dependencies (e.g., torch, transformers)..."

if command -v uv &> /dev/null; then
    cd /opt/qdrant-rag-mcp
    # Perform a full sync with all specialized model dependencies
    uv sync --no-dev --extra models --extra performance

    # Initialize a default .env if it doesn't exist
    if [ ! -f "/opt/qdrant-rag-mcp/.env" ]; then
        echo "📝 Creating default system-wide .env..."
        cp .env.example .env 2>/dev/null || touch .env
        echo "# Qdrant RAG MCP Global Configuration" > .env
        echo "QDRANT_HOST=localhost" >> .env
        echo "QDRANT_PORT=6333" >> .env
        echo "LOG_LEVEL=INFO" >> .env
        echo "SENTENCE_TRANSFORMERS_HOME=/opt/qdrant-rag-mcp/.cache/models" >> .env
    fi

    # Ensure cache directory exists and is writable
    mkdir -p /opt/qdrant-rag-mcp/.cache/models

    # Ensure all users can read and execute the environment
    echo "🔐 Setting permissions..."
    chmod -R 755 /opt/qdrant-rag-mcp
    # Allow writing to the global cache for model downloads if run as root/sudo
    chmod -R 777 /opt/qdrant-rag-mcp/.cache
else
    echo "⚠️  Warning: 'uv' not found. You will need to install it and run 'uv sync' in /opt/qdrant-rag-mcp manually."
fi

exit 0
EOF
chmod +x "$PKG_DIR/DEBIAN/postinst"

# Fix permissions (dpkg-deb is strict about DEBIAN directory permissions)
chmod -R 755 "$PKG_DIR/DEBIAN"

# Build the package
echo "🔨 Compiling .deb package..."
dpkg-deb --build "$PKG_DIR"

# Move the resulting package to the releases directory
mv "${PKG_DIR}.deb" "$RELEASE_DIR/"

echo ""
echo "✅ Build complete!"
echo "📦 Package: $RELEASE_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "To install:"
echo "  sudo dpkg -i $RELEASE_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo ""
echo "📍 Note: The installation will take a few minutes to download dependencies."
echo "   This ensures the MCP server starts instantly once installed."
