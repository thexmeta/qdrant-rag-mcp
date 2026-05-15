import os
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Import custom ONNX support
try:
    from .onnx_embeddings import ONNXSparseManager, ONNX_RUNTIME_AVAILABLE, SparseEmbeddingResult
except ImportError:
    try:
        from onnx_embeddings import ONNXSparseManager, ONNX_RUNTIME_AVAILABLE, SparseEmbeddingResult
    except ImportError:
        ONNX_RUNTIME_AVAILABLE = False
        logger.debug("onnx_embeddings not available, sparse embeddings disabled")

from dataclasses import dataclass
@dataclass
class DeduplicatedSparseEmbedding:
    """Helper class for deduplicated sparse embeddings compatible with Qdrant models.SparseVector"""
    indices: List[int]
    values: List[float]

class SparseEmbeddingsManager:
    """Manages sparse embedding generation (BM42) using fastembed or raw ONNX"""

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions", cache_dir: Optional[str] = None, sparse_method: str = "bm42"):
        self.model_name = model_name
        self.sparse_method = sparse_method
        self.cache_dir = cache_dir
        self.model = None
        self.is_custom_onnx = False

    def _get_model(self):
        if self.model is None:
            # Resolve model path
            model_path = self.model_name
            if not os.path.isdir(model_path):
                potential_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_"))
                if os.path.isdir(potential_path):
                    model_path = potential_path
                else:
                    potential_path = os.path.join(self.cache_dir, self.model_name)
                    if os.path.isdir(potential_path):
                        model_path = potential_path
            
            # Use onnxruntime if it's a local path or explicitly requested
            from config import get_config
            backend = get_config().get("embeddings.backend", "onnxruntime")
            
            if (backend == "onnxruntime" or backend == "fastembed" or os.path.isdir(model_path)) and ONNX_RUNTIME_AVAILABLE:
                logger.info(f"Loading sparse model using onnxruntime: {model_path}")
                self.model = ONNXSparseManager(model_path=model_path, cache_dir=self.cache_dir)
                self.is_custom_onnx = True
            else:
                raise ImportError("onnxruntime is required for sparse embeddings.")

        return self.model

    def encode(self, texts: Union[str, List[str]]) -> List[Any]:
        """
        Generate sparse embeddings.
        Returns a list of SparseEmbedding objects with .indices and .values
        """
        model = self._get_model()
        if isinstance(texts, str):
            texts = [texts]

        if self.is_custom_onnx:
            return model.encode(texts)
        else:
            # Enable truncation for fastembed sparse models
            raw_embeddings = list(model.embed(texts, truncate=True))
            
            # Ensure unique indices for Qdrant compatibility (v0.3.2 stabilization)
            final_embeddings = []
            for emb in raw_embeddings:
                unique_indices = {}
                for idx, val in zip(emb.indices, emb.values):
                    idx_int = int(idx)
                    unique_indices[idx_int] = unique_indices.get(idx_int, 0.0) + float(val)
                
                # Sort indices for consistency
                sorted_idx = sorted(unique_indices.keys())
                
                final_embeddings.append(DeduplicatedSparseEmbedding(
                    indices=sorted_idx,
                    values=[unique_indices[i] for i in sorted_idx]
                ))
            return final_embeddings

    def embed_document(self, text: str) -> Any:
        """Alias for encode for single document"""
        results = self.encode([text])
        return results[0] if results else None

    def embed_query(self, text: str) -> Any:
        """Alias for encode for single query"""
        results = self.encode([text])
        return results[0] if results else None

_sparse_manager = None

def get_sparse_embeddings_manager(config=None) -> Optional[SparseEmbeddingsManager]:
    global _sparse_manager
    if _sparse_manager is None:
        if config is None:
            from config import get_config
            config = get_config()

        # Check in multiple possible config locations
        sparse_model = config.get("sparse_embeddings", {}).get("model")
        if not sparse_model:
            sparse_model = config.get("hybrid_search", {}).get("sparse_model", "Qdrant/bm42-all-minilm-l6-v2-attentions")
            
        sparse_method = config.get("sparse_embeddings", {}).get("method")
        if not sparse_method:
            sparse_method = config.get("hybrid_search", {}).get("sparse_method", "bm42")
                    
        if sparse_method == "bm42":
            cache_dir = config.get("embeddings", {}).get("cache_dir", "./data/models")
            _sparse_manager = SparseEmbeddingsManager(model_name=sparse_model, cache_dir=cache_dir, sparse_method="bm42")
        else:
            return None
    return _sparse_manager
