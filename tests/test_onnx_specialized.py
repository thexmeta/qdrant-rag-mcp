import os
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.utils.embeddings import get_embeddings_manager
from src.config import Config

def test_specialized_onnx():
    config_path = project_root / "config" / "server_config.json"
    print(f"Loading config from: {config_path}")
    
    # Force specialized embeddings enabled if not already
    os.environ["QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED"] = "true"
    
    config_obj = Config(str(config_path))
    config_data = config_obj.config
    
    print("Initializing manager...")
    manager = get_embeddings_manager(config_data)
    
    content_types = ["code", "config", "documentation", "general"]
    
    for ct in content_types:
        print(f"\n--- Testing Content Type: {ct} ---")
        text = f"Sample {ct} text for ONNX validation."
        
        try:
            embedding = manager.encode(text, content_type=ct)
            print(f"Success! Dimension: {embedding.shape[1]}")
            # Try to get model info for this specific type
            # Specialized manager stores models in its internal cache
            # We can't easily get info via the unified manager for a specific type yet
        except Exception as e:
            print(f"Failed for {ct}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_specialized_onnx()
