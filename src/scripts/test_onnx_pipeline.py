# src/scripts/test_onnx_pipeline.py
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from utils.embeddings import get_embeddings_manager
from utils.sparse_embeddings import get_sparse_embeddings_manager
from utils.hybrid_search import get_hybrid_searcher

def test_pipeline():
    print("--- Testing ONNX Pipeline ---")
    
    # 1. Check Config
    config = get_config()
    backend = config.get("embeddings.backend")
    sparse_method = config.get("hybrid_search.sparse_method")
    
    print(f"Config Backend: {backend}")
    print(f"Config Sparse Method: {sparse_method}")
    
    assert backend == "onnxruntime", f"Expected onnxruntime, got {backend}"
    assert sparse_method == "bm42", f"Expected bm42, got {sparse_method}"
    
    # 2. Check Embeddings Manager
    em = get_embeddings_manager(config.config)
    print(f"Embeddings Manager: {type(em).__name__}")
    
    # Check if using FastEmbed or Specialized (which we set to use FastEmbed)
    if hasattr(em, "manager"):
        inner_manager = em.manager
        print(f"Inner Manager: {type(inner_manager).__name__}")
        
        # If it's SpecializedEmbeddingManager, check its default backend
        if hasattr(inner_manager, "default_backend"):
            print(f"Specialized Backend: {inner_manager.default_backend}")
            assert inner_manager.default_backend == "onnxruntime"
    
    # 3. Check Sparse Embeddings Manager
    sm = get_sparse_embeddings_manager(config.config)
    print(f"Sparse Manager: {type(sm).__name__}")
    assert sm.sparse_method == "bm42"
    
    # 4. Test Search Logic Integration
    # We can't easily test the full qdrant search without a running instance,
    # but we can check if the hybrid searcher has the new method.
    hs = get_hybrid_searcher()
    assert hasattr(hs, "sparse_search_qdrant"), "HybridSearcher missing sparse_search_qdrant method"
    print("HybridSearcher has sparse_search_qdrant method.")
    
    print("\n--- Pipeline Check Passed! ---")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline Check Failed: {e}")
        sys.exit(1)
