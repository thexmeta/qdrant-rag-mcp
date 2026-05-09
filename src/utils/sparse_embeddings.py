import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    logger.debug("fastembed not available, sparse embeddings (BM42) disabled.")

class SparseEmbeddingsManager:
    """Manages sparse embedding generation (BM42) using fastembed"""

    def __init__(self, model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None

    def _get_model(self):
        if self.model is None:
            if not FASTEMBED_AVAILABLE:
                raise ImportError("fastembed is required for sparse embeddings. Install it with `pip install fastembed`.")

            logger.info(f"Loading sparse embedding model: {self.model_name}")
            self.model = SparseTextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)

        return self.model

    def encode(self, texts: Union[str, List[str]]) -> List[Any]:
        """
        Generate sparse embeddings.
        Returns a list of SparseEmbedding objects with .indices and .values
        """
        model = self._get_model()
        if isinstance(texts, str):
            texts = [texts]

        embeddings = list(model.embed(texts))
        return embeddings

_sparse_manager = None

def get_sparse_embeddings_manager(config=None) -> Optional[SparseEmbeddingsManager]:
    global _sparse_manager
    if _sparse_manager is None:
        if config is None:
            from ..config import get_config
            config = get_config()

        sparse_method = config.get("hybrid_search.sparse_method", "bm25")
        if sparse_method == "bm42":
            if not FASTEMBED_AVAILABLE:
                logger.warning("BM42 is enabled but fastembed is not installed. Falling back to bm25 local approach.")
                return None
            model_name = config.get("hybrid_search.sparse_model", "Qdrant/bm42-all-minilm-l6-v2-attentions")
            cache_dir = config.get("embeddings.cache_dir", "./data/models")
            _sparse_manager = SparseEmbeddingsManager(model_name=model_name, cache_dir=cache_dir)
        else:
            return None
    return _sparse_manager
