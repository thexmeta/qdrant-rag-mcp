#!/bin/bash
# Runner for the Qdrant RAG MCP HTTP REST API
# This allows testing the RAG server functionality via HTTP requests

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# The application automatically prioritizes config/server_config.json
# as defined in the core configuration logic.

# Run the HTTP server using uv
# Note: port 8081 is the default in src/http_server.py
echo "🚀 Starting Qdrant RAG HTTP Server on port 8081..."
echo "📍 Using configuration from: $PROJECT_ROOT/config/server_config.json"
exec uv run python "$PROJECT_ROOT/src/http_server.py" "$@"
