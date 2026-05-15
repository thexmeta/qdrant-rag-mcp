import os
import sys
import asyncio
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

import pytest
from src.utils.logging import get_project_logger

@pytest.mark.asyncio
async def test_indexing(monkeypatch):
    # Force local model path for integration test
    model_path = "./data/models/qdrant_all_miniLM_L6_v2_with_attentions"
    monkeypatch.setenv("QDRANT_SPARSE_MODEL", model_path)
    
    # Reset global managers to pick up new config
    import src.utils.embeddings
    import src.utils.sparse_embeddings
    import src.config
    import src.qdrant_mcp_context_aware
    
    # Try both with and without src. prefix just in case
    for mod in [src.utils.embeddings, src.utils.sparse_embeddings, src.config, src.qdrant_mcp_context_aware]:
        if hasattr(mod, "_embeddings_manager"): mod._embeddings_manager = None
        if hasattr(mod, "_sparse_manager"): mod._sparse_manager = None
        if hasattr(mod, "_config"): mod._config = None
        
    # Also try the versions without src. prefix if they exist in sys.modules
    import sys
    for name in ['utils.embeddings', 'utils.sparse_embeddings', 'config', 'qdrant_mcp_context_aware']:
        if name in sys.modules:
            mod = sys.modules[name]
            if hasattr(mod, "_embeddings_manager"): mod._embeddings_manager = None
            if hasattr(mod, "_sparse_manager"): mod._sparse_manager = None
            if hasattr(mod, "_config"): mod._config = None
    
    from src.qdrant_mcp_context_aware import index_documentation
    
    test_file = str(project_root / "tests" / "onnx_test.md")
    print(f"Indexing file: {test_file}")
    
    # Run indexing
    result = index_documentation(test_file)
    
    print(f"Indexing result: {result}")
    
    if "error" in result:
        print(f"FAILED: {result['error']}")
        assert "error" not in result, f"Indexing failed: {result['error']}\nDetails: {result.get('details')}"
    else:
        print("SUCCESS: File indexed with ONNX backend!")

if __name__ == "__main__":
    asyncio.run(test_indexing())
