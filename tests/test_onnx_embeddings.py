import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.utils.embeddings import get_embeddings_manager
from src.config import get_config

def test_fastembed_backend():
    # Force fastembed backend in config
    config = {
        "embeddings": {
            "backend": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
            "cache_dir": "./data/models"
        }
    }
    
    print("Initializing manager with fastembed backend...")
    manager = get_embeddings_manager(config)
    
    print(f"Manager type: {type(manager)}")
    print(f"Model info: {manager.get_model_info()}")
    
    text = "This is a test of the ONNX backend."
    print(f"Encoding text: '{text}'")
    
    embedding = manager.encode(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 elements: {embedding[0][:5]}")
    
    assert embedding.shape == (1, 384)  # BGE small dimension is 384
    print("✓ FastEmbed backend test passed!")

if __name__ == "__main__":
    try:
        import fastembed
        print(f"FastEmbed version: {fastembed.__version__}")
        test_fastembed_backend()
    except ImportError as e:
        print(f"ImportError: {e}")
        print("FastEmbed not installed, skipping test.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
