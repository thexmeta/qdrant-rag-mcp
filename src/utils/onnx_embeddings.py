import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import onnxruntime and tokenizers
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    logger.debug("onnxruntime or tokenizers not available, custom ONNX support disabled")


@dataclass
class SparseEmbeddingResult:
    """Standardized sparse embedding result compatible with Qdrant SparseVector"""
    indices: List[int]
    values: List[float]


class ONNXBaseManager:
    """Base class for ONNX-based model managers"""
    
    def __init__(
        self,
        model_path: str,
        threads: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        self.cache_dir = cache_dir or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.original_model_path = model_path
        self.model_path = self._resolve_model_path(model_path)
        self.model_name = os.path.basename(self.model_path.rstrip("/"))
        self.threads = threads
        self._session = None
        self._tokenizer = None

    def _resolve_model_path(self, model_name: str) -> str:
        """
        Resolve model name to local path.
        Supports:
        - Local directory paths
        - HuggingFace model names (checks local cache or ./data/models)
        """
        # Explicit mapping for common model names to local directories
        name_mapping = {
            "all-MiniLM-L6-v2": "./data/models/qdrant_all_miniLM_L6_v2_with_attentions",
            "sentence-transformers/all-MiniLM-L6-v2": "./data/models/qdrant_all_miniLM_L6_v2_with_attentions",
            "BAAI/bge-small-en-v1.5": "./data/models/qdrant--bge-small-en-v1.5-onnx-q",
            "nomic-ai/CodeRankEmbed": "./data/models/stella_en_400M_v5/int8", # Stella is often used as a substitute
            "jinaai/jina-embeddings-v3": "./data/models/jina-reranker-v3-onnx-int8-NG"
        }
        
        if model_name in name_mapping:
            mapped_path = name_mapping[model_name]
            if os.path.isdir(mapped_path):
                return mapped_path

        # Check if it's already a local directory
        if os.path.isdir(model_name):
            return model_name
        
        # Check common cache directories
        cache_dirs = [self.cache_dir, "./data/models", "./models"]
        
        # Clean model name for path matching
        clean_name = model_name.replace("/", "--").replace("sentence-transformers--", "")
        
        for base_dir in cache_dirs:
            if not base_dir or not os.path.exists(base_dir):
                continue
                
            # Try various naming variants
            variants = [
                model_name,
                model_name.replace("/", "_"),
                model_name.replace("/", "--"),
                f"models--{model_name.replace('/', '--')}",
                clean_name
            ]
            
            for variant in variants:
                potential = os.path.join(base_dir, variant)
                # Check for direct match or deep nested hub structure
                if os.path.isdir(potential):
                    # Check if it's a hub snapshot directory
                    snapshots_dir = os.path.join(potential, "snapshots")
                    if os.path.isdir(snapshots_dir):
                        snapshots = os.listdir(snapshots_dir)
                        if snapshots:
                            potential = os.path.join(snapshots_dir, snapshots[0])
                    
                    # Verify it has model.onnx or similar
                    if os.path.exists(os.path.join(potential, "model.onnx")) or \
                       os.path.exists(os.path.join(potential, "onnx", "model.onnx")) or \
                       os.path.exists(os.path.join(potential, "int8", "model.onnx")):
                        return potential

        # Last resort: check if any of the variants exist in ./data/models directly
        for variant in [model_name, clean_name]:
            test_path = os.path.join("./data/models", variant)
            if os.path.isdir(test_path):
                return test_path

        # If not found locally, we'll let the session property raise an error or we could try downloading
        logger.warning(f"Model {model_name} not found in cache. Using as raw path.")
        return model_name
        
    @property
    def session(self) -> Any:
        """Lazy load the ONNX session"""
        if self._session is None:
            if not ONNX_RUNTIME_AVAILABLE:
                raise ImportError("onnxruntime is required for ONNX managers.")
            
            # Check for model.onnx in the directory
            if os.path.isfile(self.model_path) and self.model_path.endswith(".onnx"):
                onnx_file = self.model_path
            else:
                onnx_file = os.path.join(self.model_path, "model.onnx")
                if not os.path.exists(onnx_file):
                    # Try searching in subdirectories (like int8/, fp16/, q4/, raw/)
                    found = False
                    for sub in ["int8", "fp16", "q4", "raw", "onnx"]:
                        test_path = os.path.join(self.model_path, sub, "model.onnx")
                        if os.path.exists(test_path):
                            onnx_file = test_path
                            found = True
                            break
                    if not found:
                        # Try searching for ANY .onnx file in the directory
                        import glob
                        onnx_files = glob.glob(os.path.join(self.model_path, "**/*.onnx"), recursive=True)
                        if onnx_files:
                            onnx_file = onnx_files[0]
                            logger.info(f"Auto-discovered ONNX file: {onnx_file}")
                        else:
                            raise FileNotFoundError(f"ONNX model file not found in {self.model_path}")
            
            # Set session options
            sess_options = ort.SessionOptions()
            if self.threads:
                sess_options.intra_op_num_threads = self.threads
            
            logger.info(f"Loading ONNX session from: {onnx_file}")
            self._session = ort.InferenceSession(onnx_file, sess_options)
            
        return self._session

    @property
    def tokenizer(self) -> Any:
        """Lazy load the tokenizer"""
        if self._tokenizer is None:
            if not ONNX_RUNTIME_AVAILABLE:
                raise ImportError("tokenizers is required for ONNX managers.")
            
            # Look for tokenizer.json in the same directory as the model or its parent
            # Handle both model file path and directory path
            sess_path = self.session._model_path if hasattr(self.session, '_model_path') else self.model_path
            model_dir = os.path.dirname(sess_path) if os.path.isfile(sess_path) else sess_path
            
            tokenizer_file = os.path.join(model_dir, "tokenizer.json")
            
            if not os.path.exists(tokenizer_file):
                # Try the base model path
                tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
            
            if not os.path.exists(tokenizer_file):
                # Try recursive search up one level
                parent_dir = os.path.dirname(self.model_path)
                tokenizer_file = os.path.join(parent_dir, "tokenizer.json")

            if not os.path.exists(tokenizer_file):
                # Check for tokenizer_config.json if tokenizer.json is missing? 
                # No, raw tokenizers library needs tokenizer.json
                raise FileNotFoundError(f"Tokenizer file not found for {self.model_path}")
            
            logger.info(f"Loading tokenizer from: {tokenizer_file}")
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
            # Enable truncation to 512 tokens by default to prevent ONNX errors
            self._tokenizer.enable_truncation(max_length=512)
            
        return self._tokenizer

    def _prepare_inputs(self, encoding) -> Dict[str, np.ndarray]:
        """Prepare inputs for the ONNX session based on available input names"""
        input_names = [i.name for i in self.session.get_inputs()]
        inputs = {}
        
        if "input_ids" in input_names:
            inputs["input_ids"] = np.array([encoding.ids], dtype=np.int64)
        if "attention_mask" in input_names:
            inputs["attention_mask"] = np.array([encoding.attention_mask], dtype=np.int64)
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
            
        return inputs


class ONNXDenseManager(ONNXBaseManager):
    """Manages dense embedding generation using raw onnxruntime"""

    def __init__(self, model_path: str, threads: Optional[int] = None, **kwargs):
        super().__init__(model_path, threads, **kwargs)
        self._model_dimension = None

    @property
    def dimension(self) -> int:
        if self._model_dimension is None:
            outputs = self.session.get_outputs()
            for output in outputs:
                if "embedding" in output.name or "last_hidden_state" in output.name:
                    if len(output.shape) == 2:
                        self._model_dimension = output.shape[1]
                        break
                    elif len(output.shape) == 3:
                        self._model_dimension = output.shape[2]
                        break
            
            if self._model_dimension is None:
                self._model_dimension = outputs[0].shape[-1]
                
        return self._model_dimension

    def encode(self, texts: Union[str, List[str]], content_type: Optional[str] = None, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            encoding = self.tokenizer.encode(text)
            inputs = self._prepare_inputs(encoding)
            
            output_names = [o.name for o in self.session.get_outputs()]
            output_name = next((n for n in output_names if "embedding" in n or "last_hidden_state" in n), output_names[0])
            
            res = self.session.run([output_name], inputs)
            
            # Simple mean pooling if it's last_hidden_state (batch, seq, dim)
            out = res[0]
            if len(out.shape) == 3:
                attention_mask = np.array(encoding.attention_mask)
                mask = np.expand_dims(attention_mask, -1)
                masked_out = out[0] * mask
                # Avoid division by zero
                mask_sum = mask.sum(axis=0)
                mask_sum[mask_sum == 0] = 1
                pooled = masked_out.sum(axis=0) / mask_sum
                embeddings.append(pooled)
            else:
                embeddings.append(out[0])
            
        return np.array(embeddings, dtype=np.float32)

    def get_model_info(self) -> Dict[str, Any]:
        try:
            dim = self.dimension
        except Exception as e:
            logger.warning(f"Could not determine dimension for {self.model_path}: {e}")
            dim = None
            
        return {
            "model_path": self.model_path,
            "dimension": dim,
            "backend": "onnxruntime (Dense)",
            "threads": self.threads,
            "is_loaded": self._session is not None
        }


class ONNXRerankerManager(ONNXBaseManager):
    """Manages reranking using raw onnxruntime cross-encoders"""

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rerank documents against a query.
        Returns a list of (index, score) sorted by score.
        """
        scores = []
        for doc in documents:
            encoding = self.tokenizer.encode(query, doc)
            inputs = self._prepare_inputs(encoding)
            
            res = self.session.run(None, inputs)
            # Logit is usually the first element of the first output
            # Handle (batch, 1) or (batch, seq, 1)
            out = res[0]
            if len(out.shape) == 3:
                # Take CLS logit
                score = float(out[0][0][0])
            else:
                score = float(out[0][0])
            scores.append(score)
            
        # Sort indices by score
        ranked_indices = np.argsort(scores)[::-1]
        results = [(int(i), float(scores[i])) for i in ranked_indices]
        
        if top_k:
            results = results[:top_k]
            
        return results

class ONNXSparseManager(ONNXBaseManager):
    """Manages sparse embedding generation (BM42 style) using raw onnxruntime"""

    def encode(self, texts: Union[str, List[str]], content_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Generate pseudo-sparse embeddings using attention weights if available"""
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        for text in texts:
            encoding = self.tokenizer.encode(text)
            inputs = self._prepare_inputs(encoding)
            
            # Try to get attention outputs
            output_names = [o.name for o in self.session.get_outputs()]
            attention_outputs = [n for n in output_names if "attention" in n]
            
            if not attention_outputs:
                # Fallback: use uniform weights for non-special tokens
                indices = np.array(encoding.ids, dtype=np.int64).tolist()
                attention_mask = np.array(encoding.attention_mask, dtype=np.float32).tolist()
                
                # Aggregate duplicate indices
                unique_indices = {}
                for idx, val in zip(indices, attention_mask):
                    if val > 0:
                        idx_item = int(idx)
                        unique_indices[idx_item] = unique_indices.get(idx_item, 0.0) + float(val)
                
                sorted_indices = sorted(unique_indices.keys())
                final_values = [unique_indices[idx] for idx in sorted_indices]
                
                results.append(SparseEmbeddingResult(indices=sorted_indices, values=final_values))
                continue
                
            res = self.session.run(attention_outputs, inputs)
            
            # Combine attention weights across layers and heads
            combined_weights = np.zeros(len(encoding.ids), dtype=np.float32)
            for layer_att in res:
                # layer_att[0] is (heads, seq, seq)
                combined_weights += layer_att[0].sum(axis=(0, 1))
                
            indices = np.array(encoding.ids, dtype=np.int64)
            attention_mask = np.array(encoding.attention_mask, dtype=np.int64)
            mask = attention_mask > 0
            
            indices = indices[mask]
            weights = combined_weights[mask]
            
            # Aggregate duplicate indices (sum values)
            unique_indices = {}
            for idx, val in zip(indices, weights):
                idx_item = int(idx)
                unique_indices[idx_item] = unique_indices.get(idx_item, 0.0) + float(val)
            
            sorted_indices = sorted(unique_indices.keys())
            final_values = [unique_indices[idx] for idx in sorted_indices]
            
            # Normalize final values
            max_val = max(final_values) if final_values else 0
            if max_val > 0:
                final_values = [v / max_val for v in final_values]
                
            results.append(SparseEmbeddingResult(
                indices=sorted_indices,
                values=final_values
            ))
            
        return results

# Legacy alias
ONNXRuntimeManager = ONNXDenseManager
