# tests/unit/test_onnx_bm42_stabilization.py
import pytest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import get_config
from utils.embeddings import get_embeddings_manager, UnifiedEmbeddingsManager
from utils.sparse_embeddings import get_sparse_embeddings_manager, SparseEmbeddingsManager
from utils.hybrid_search import get_hybrid_searcher

def test_config_defaults_for_onnx_bm42():
    """Verify that configuration defaults are correctly set for ONNX and BM42"""
    config = get_config()
    
    # Check embeddings backend
    assert config.get("embeddings.backend") == "onnxruntime"
    
    # Check sparse search method
    assert config.get("hybrid_search.sparse_method") == "bm42"
    assert config.get("hybrid_search.sparse_model") == "Qdrant/bm42-all-minilm-l6-v2-attentions"

def test_unified_embeddings_manager_backend_selection():
    """Verify that UnifiedEmbeddingsManager correctly selects ONNX backend"""
    config = {
        "embeddings": {
            "backend": "onnxruntime",
            "model": "all-MiniLM-L6-v2"
        }
    }
    
    # Disable specialized embeddings for this test and reset global state
    with patch('utils.embeddings._use_specialized', False), \
         patch('utils.embeddings.ONNX_RUNTIME_AVAILABLE', True):
        
        with patch('utils.embeddings.ONNXRuntimeManager') as mock_onnx_mgr:
            # Mock ONNXRuntimeManager to avoid loading real models
            mock_mgr_instance = MagicMock()
            mock_onnx_mgr.return_value = mock_mgr_instance
            
            manager = UnifiedEmbeddingsManager(config)
            
            assert not manager.use_specialized
            assert manager.config.get("embeddings", {}).get("backend") == "onnxruntime"
            # The manager should be a ONNXRuntimeManager
            assert manager.manager == mock_mgr_instance

def test_sparse_embeddings_manager_initialization():
    """Verify that SparseEmbeddingsManager initializes with BM42 by default"""
    config = {
        "hybrid_search": {
            "sparse_method": "bm42"
        }
    }
    
    with patch('utils.sparse_embeddings.ONNXSparseManager') as mock_onnx_mgr:
        mock_onnx_mgr.return_value = MagicMock()
        
        # Reset the global manager before test
        import utils.sparse_embeddings
        utils.sparse_embeddings._sparse_manager = None
        
        manager = get_sparse_embeddings_manager(config)
        
        assert manager is not None
        assert manager.sparse_method == "bm42"
        # The model name should be derived from the hybrid_search config or default
        # (It defaults to Qdrant/bm42-all-minilm-l6-v2-attentions in get_sparse_embeddings_manager)
        assert "bm42" in manager.model_name.lower()

def test_hybrid_searcher_qdrant_sparse_method():
    """Verify that HybridSearcher has the new native sparse search method"""
    searcher = get_hybrid_searcher()
    assert hasattr(searcher, "sparse_search_qdrant")

@pytest.mark.asyncio
async def test_sparse_search_qdrant_logic():
    """Test the logic of sparse_search_qdrant with mocks"""
    searcher = get_hybrid_searcher()
    
    mock_client = MagicMock()
    mock_sparse_manager = MagicMock()
    
    # Mock sparse embedding generation
    mock_emb = MagicMock()
    mock_emb.indices = [1, 2, 3]
    mock_emb.values = [0.1, 0.2, 0.3]
    mock_sparse_manager.embed_query.return_value = mock_emb
    
    # Mock Qdrant results
    mock_result = MagicMock()
    mock_result.score = 0.95
    mock_result.payload = {"file_path": "test.py", "chunk_index": 0}
    
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_result]
    mock_client.query_points.return_value = mock_query_response
    
    results = searcher.sparse_search_qdrant(
        qdrant_client=mock_client,
        collection_name="test_collection",
        query="test query",
        sparse_manager=mock_sparse_manager,
        k=5
    )
    
    assert len(results) == 1
    assert results[0][0] == "test.py_0"
    assert results[0][1] == 0.95
    
    # Verify client call
    mock_client.query_points.assert_called_once()
    args, kwargs = mock_client.query_points.call_args
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["using"] == "sparse"
    assert kwargs["limit"] == 5

def test_fusion_logic_with_bm42_results():
    """Verify that hybrid searcher correctly fuses vector and sparse results"""
    searcher = get_hybrid_searcher()
    
    query = "test query"
    vector_results = [("doc1", 0.8), ("doc2", 0.7)]
    # Use bm25_results parameter for sparse results
    sparse_results = [("doc2", 0.9), ("doc3", 0.6)]
    
    weights = {"vector": 0.5, "bm25": 0.5}
    
    fused = searcher.linear_combination_with_exact_match(
        vector_results=vector_results,
        bm25_results=sparse_results,
        query=query,
        result_objects_map={"doc1": MagicMock(), "doc2": MagicMock(), "doc3": MagicMock()},
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    assert len(fused) > 0
    # doc2 should be ranked high as it's in both
    doc2_result = next(r for r in fused if r.content == "doc2")
    doc1_result = next(r for r in fused if r.content == "doc1")
    
    # Check combined_score
    assert doc2_result.combined_score > doc1_result.combined_score
