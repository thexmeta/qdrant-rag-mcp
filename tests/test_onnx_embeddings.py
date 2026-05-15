import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.utils.embeddings import get_embeddings_manager
from src.config import get_config

def test_onnxruntime_backend():
    # Force onnxruntime backend in config with local model path
    model_path = "./data/models/qdrant_all_miniLM_L6_v2_with_attentions"
    config = {
        "embeddings": {
            "backend": "onnxruntime",
            "model": model_path,
            "cache_dir": "./data/models"
        }
    }
    
    print(f"Initializing manager with onnxruntime backend and model: {model_path}")
    manager = get_embeddings_manager(config)
    
    print(f"Manager type: {type(manager)}")
    print(f"Model info: {manager.get_model_info()}")
    
    text = "This is a test of the ONNX backend."
    print(f"Encoding text: '{text}'")
    
    embedding = manager.encode(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 elements: {embedding[0][:5]}")
    
    # qdrant_all_miniLM_L6_v2 dimension is 384
    assert embedding.shape == (1, 384)
    print("✓ ONNX Runtime backend test passed!")

if __name__ == "__main__":
    try:
        from src.utils.onnx_embeddings import ONNX_RUNTIME_AVAILABLE
        if ONNX_RUNTIME_AVAILABLE:
            test_onnxruntime_backend()
        else:
            print("onnxruntime not available, skipping test.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
