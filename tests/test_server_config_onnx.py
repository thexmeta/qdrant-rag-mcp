import os
import sys
import json
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.utils.embeddings import get_embeddings_manager
from src.config import Config

def test_server_config():
    config_path = project_root / "config" / "server_config.json"
    print(f"Loading config from: {config_path}")
    
    # Use the Config class to load and resolve env vars
    config_obj = Config(str(config_path))
    config_data = config_obj.config
    
    print("Initializing manager with resolved config...")
    manager = get_embeddings_manager(config_data)
    
    print(f"Manager type: {type(manager)}")
    print(f"Model info: {manager.get_model_info()}")
    
    # Check if it's using the model from config
    model_name = config_data['embeddings']['model']
    print(f"Expected model: {model_name}")
    
    text = "Testing with Stella ONNX model from server_config.json"
    print(f"Encoding text: '{text}'")
    
    try:
        embedding = manager.encode(text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Success! Indexed with Stella ONNX.")
    except Exception as e:
        print(f"Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_server_config()
