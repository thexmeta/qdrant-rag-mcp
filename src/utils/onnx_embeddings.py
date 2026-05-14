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

# Import fastembed
try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    logger.debug("fastembed not available, ONNX dense/sparse embeddings disabled")


class ONNXBaseManager:
    """Base class for ONNX-based model managers"""
    
    def __init__(
        self,
        model_path: str,
        threads: Optional[int] = None,
        **kwargs
    ):
        self.model_path = model_path
        self.threads = threads
        self._session = None
        self._tokenizer = None
        
    @property
    def session(self) -> Any:
        """Lazy load the ONNX session"""
        if self._session is None:
            if not ONNX_RUNTIME_AVAILABLE:
                raise ImportError("onnxruntime is required for ONNX managers.")
            
            # Check for model.onnx in the directory
            onnx_file = os.path.join(self.model_path, "model.onnx")
            if not os.path.exists(onnx_file):
                # Try searching in subdirectories (like int8/ or fp16/)
                found = False
                for sub in ["int8", "fp16", "q4"]:
                    test_path = os.path.join(self.model_path, sub, "model.onnx")
                    if os.path.exists(test_path):
                        onnx_file = test_path
                        found = True
                        break
                if not found:
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

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
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
        return {
            "model_path": self.model_path,
            "dimension": self.dimension,
            "backend": "onnxruntime (Dense)",
            "threads": self.threads
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


@dataclass
class SparseEmbedding:
    indices: np.ndarray
    values: np.ndarray


class ONNXSparseManager(ONNXBaseManager):
    """Manages sparse embedding generation (BM42 style) using raw onnxruntime"""

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[SparseEmbedding]:
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
                indices = np.array(encoding.ids, dtype=np.int64)
                attention_mask = np.array(encoding.attention_mask, dtype=np.float32)
                # Filter out padding
                mask = attention_mask > 0
                results.append(SparseEmbedding(indices[mask], attention_mask[mask]))
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
            
            # Normalize values
            values = combined_weights[mask]
            if values.max() > 0:
                values = values / values.max()
                
            results.append(SparseEmbedding(indices[mask], values))
            
        return results

# Legacy alias
ONNXRuntimeManager = ONNXDenseManager
