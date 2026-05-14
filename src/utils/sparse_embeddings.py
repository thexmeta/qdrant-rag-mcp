import os
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Import custom ONNX support
try:
    from .onnx_embeddings import ONNXSparseManager, ONNX_RUNTIME_AVAILABLE, FASTEMBED_AVAILABLE, SparseTextEmbedding
except ImportError:
    try:
        from onnx_embeddings import ONNXSparseManager, ONNX_RUNTIME_AVAILABLE, FASTEMBED_AVAILABLE, SparseTextEmbedding
    except ImportError:
        ONNX_RUNTIME_AVAILABLE = False
        FASTEMBED_AVAILABLE = False
        logger.debug("onnx_embeddings not available, sparse embeddings falling back to basic checks")

class SparseEmbeddingsManager:
    """Manages sparse embedding generation (BM42) using fastembed or raw ONNX"""

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.is_custom_onnx = False

    def _get_model(self):
        if self.model is None:
            # Check if it's a local path
            is_local_path = os.path.isdir(self.model_name)
            
            if is_local_path and ONNX_RUNTIME_AVAILABLE:
                logger.info(f"Loading sparse model from local path: {self.model_name}")
                self.model = ONNXSparseManager(model_path=self.model_name)
                self.is_custom_onnx = True
            elif FASTEMBED_AVAILABLE:
                logger.info(f"Loading sparse embedding model: {self.model_name}")
                self.model = SparseTextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
                self.is_custom_onnx = False
            else:
                raise ImportError("fastembed or onnxruntime is required for sparse embeddings.")

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
            embeddings = list(model.embed(texts))
            return embeddings

_sparse_manager = None

def get_sparse_embeddings_manager(config=None) -> Optional[SparseEmbeddingsManager]:
    global _sparse_manager
    if _sparse_manager is None:
        if config is None:
            from .config import get_config
            config = get_config()

        # Check in multiple possible config locations
        sparse_model = config.get("sparse_embeddings", {}).get("model")
        if not sparse_model:
            sparse_model = config.get("hybrid_search", {}).get("sparse_model", "Qdrant/bm42-all-minilm-l6-v2-attentions")
            
        sparse_method = config.get("sparse_embeddings", {}).get("method")
        if not sparse_method:
            sparse_method = config.get("hybrid_search", {}).get("sparse_method", "bm25")
            
        if sparse_method == "bm42":
            cache_dir = config.get("embeddings", {}).get("cache_dir", "./data/models")
            _sparse_manager = SparseEmbeddingsManager(model_name=sparse_model, cache_dir=cache_dir)
        else:
            return None
    return _sparse_manager
