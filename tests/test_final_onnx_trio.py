import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.utils.embeddings import get_embeddings_manager
from src.utils.sparse_embeddings import get_sparse_embeddings_manager
from src.utils.hybrid_search import get_hybrid_searcher
from src.config import Config

def test_onnx_trio():
    config_path = project_root / "config" / "server_config.json"
    print(f"Loading config from: {config_path}")
    
    config_obj = Config(str(config_path))
    config_data = config_obj.config
    
    print("\n1. Testing Dense ONNX (Stella)...")
    dense_manager = get_embeddings_manager(config_data)
    dense_emb = dense_manager.encode("Test dense embedding")
    print(f"Dense Success! Shape: {dense_emb.shape}")
    
    print("\n2. Testing Sparse ONNX (MiniLM BM42)...")
    # BM42 requires "hybrid_search.sparse_method" to be "bm42" in config
    # Our config has it in "sparse_embeddings.method"
    sparse_manager = get_sparse_embeddings_manager(config_data)
    if sparse_manager:
        sparse_emb = sparse_manager.encode("Test sparse embedding")
        print(f"Sparse Success! Indices count: {len(sparse_emb[0].indices)}")
    else:
        print("Sparse manager not initialized (check method in config)")
    
    print("\n3. Testing Reranker ONNX (Nemotron)...")
    hybrid_searcher = get_hybrid_searcher()
    if hybrid_searcher.onnx_reranker:
        query = "What is ONNX?"
        docs = [
            "ONNX is an open format to represent machine learning models.",
            "I like pizza with extra cheese.",
            "The quick brown fox jumps over the lazy dog."
        ]
        results = hybrid_searcher.onnx_reranker.rerank(query, docs)
        print("Reranker Success!")
        for i, score in results:
            print(f"  Score: {score:.4f} | Doc: {docs[i][:50]}...")
    else:
        print("Reranker not initialized (check path in config)")

if __name__ == "__main__":
    test_onnx_trio()
