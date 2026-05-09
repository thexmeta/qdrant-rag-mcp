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
        set -a
        source "$LOC"
        set +a
        break
    fi
done

# Check for uv
UV_CMD="uv"
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    fi
fi

# Run the server
# We use the venv python directly to completely bypass 'uv run' and its sync checks.
# This ensures that regular users can run the server even if /opt is root-owned.
export PYTHONPATH="/opt/qdrant-rag-mcp/src:$PYTHONPATH"
exec /opt/qdrant-rag-mcp/.venv/bin/python /opt/qdrant-rag-mcp/src/qdrant_mcp_context_aware.py "$@"
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
echo "📦 This may take a few minutes as it downloads large dependencies..."

# Try to find uv in common locations
UV_CMD="uv"
if ! command -v uv &> /dev/null; then
    # Check common install locations
    POSSIBLE_UV=(
        "/usr/local/bin/uv"
        "/usr/bin/uv"
        "$HOME/.local/bin/uv"
        "/root/.local/bin/uv"
    )
    for loc in "${POSSIBLE_UV[@]}"; do
        if [ -f "$loc" ]; then
            UV_CMD="$loc"
            break
        fi
    done
fi

if command -v "$UV_CMD" &> /dev/null; then
    cd /opt/qdrant-rag-mcp

    # Initialize a default .env if it doesn't exist
    if [ ! -f ".env" ]; then
        echo "📝 Creating default system-wide .env..."
        cp .env.example .env 2>/dev/null || touch .env
        {
            echo "# Qdrant RAG MCP Global Configuration"
            echo "QDRANT_HOST=localhost"
            echo "QDRANT_PORT=6333"
            echo "LOG_LEVEL=INFO"
            echo "SENTENCE_TRANSFORMERS_HOME=/opt/qdrant-rag-mcp/.cache/models"
        } > .env
    fi

    # Sync environment - CRITICAL: use --no-editable to avoid permission issues later
    echo "🔄 Running uv sync..."
    "$UV_CMD" sync --no-dev --no-editable --extra models --extra performance

    # Also install the project itself via pip to ensure it's not editable
    echo "📦 Installing project as a regular package..."
    "$UV_CMD" pip install .

    # Clean up any leftover editable implementations or .pth files that cause permission errors
    echo "🧹 Cleaning up editable hooks..."
    find .venv -name "*_editable_impl_*.pth" -delete 2>/dev/null || true
    find .venv -name "*.pth" -exec grep -l "editable" {} + | xargs rm -f 2>/dev/null || true

    # Ensure cache directory exists
    mkdir -p /opt/qdrant-rag-mcp/.cache/models

    # Set permissions
    echo "🔐 Setting permissions..."
    # Make everything in /opt/qdrant-rag-mcp owned by root but readable by all
    chown -R root:root /opt/qdrant-rag-mcp
    chmod -R 755 /opt/qdrant-rag-mcp

    # Allow writing to cache and logs (if any)
    chmod -R 777 /opt/qdrant-rag-mcp/.cache

    echo "✅ Environment initialized successfully"
else
    echo "⚠️  Warning: 'uv' not found. Please install it and run 'uv sync' in /opt/qdrant-rag-mcp manually."
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
