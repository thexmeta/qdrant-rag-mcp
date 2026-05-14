import os
import sys
import asyncio
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.qdrant_mcp_context_aware import index_documentation
from src.utils.logging import get_project_logger

async def test_indexing():
    test_file = str(project_root / "tests" / "onnx_test.md")
    print(f"Indexing file: {test_file}")
    
    # Run indexing
    # index_documentation is a tool, so it's a regular function in this server
    result = index_documentation(test_file)
    
    print(f"Indexing result: {result}")
    
    if "error" in result:
        print(f"FAILED: {result['error']}")
        sys.exit(1)
    else:
        print("SUCCESS: File indexed with ONNX backend!")

if __name__ == "__main__":
    asyncio.run(test_indexing())
