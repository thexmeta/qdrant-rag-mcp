import os
import sys
import gc
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def get_system_memory() -> Optional[Dict[str, float]]:
    """Get system memory usage (supports Linux and macOS)"""
    try:
        if platform.system() == "Darwin":
            # macOS implementation using vm_stat
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            # Parse page size
            page_size_line = lines[0]
            page_size = int(page_size_line.split()[-2])
            
            # Parse memory stats
            stats = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().replace(' ', '_')
                    value = value.strip().rstrip('.')
                    if value.isdigit():
                        stats[key] = int(value) * page_size / (1024**3)  # Convert to GB
            
            # Total memory on macOS
            total_result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            total_gb = int(total_result.stdout.strip()) / (1024**3)
            
            available = stats.get('Pages_free', 0) + stats.get('Pages_inactive', 0) + stats.get('Pages_speculative', 0)
            used = total_gb - available
            
            return {
                'total_gb': total_gb,
                'used_gb': used,
                'available_gb': available,
                'wired_gb': stats.get('Pages_wired_down', 0),
                'compressed_gb': stats.get('Pages_occupied_by_compressor', 0)
            }
        else:
            # Linux implementation reading /proc/meminfo
            stats = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    key, value = line.split(':', 1)
                    # Value is in kB
                    val_kb = int(value.strip().split()[0])
                    stats[key.strip()] = val_kb / (1024**2)  # Convert to GB
            
            total_gb = stats.get('MemTotal', 0)
            available = stats.get('MemAvailable', stats.get('MemFree', 0))
            used = total_gb - available
            
            return {
                'total_gb': total_gb,
                'used_gb': used,
                'available_gb': available,
                'wired_gb': stats.get('Active', 0), # Approximate wired with Active
                'compressed_gb': stats.get('Shmem', 0) # Use Shmem as a proxy for overhead
            }
    except Exception as e:
        print(f"Error getting system memory: {e}")
        return None

def format_system_memory(info, label):
    """Format system memory info"""
    if not info:
        return f"{label}: Unable to get system memory info"
    
    return (f"{label}:\n"
            f"  Total:      {info['total_gb']:.2f} GB\n"
            f"  Used:       {info['used_gb']:.2f} GB\n"
            f"  Available:  {info['available_gb']:.2f} GB")

if __name__ == "__main__":
    print(f"Testing CodeRankEmbed memory usage on {platform.system()}...\n")

    # Set environment - use project-relative paths
    base_dir = Path(__file__).parent.parent.parent
    cache_dir = str((base_dir / "data" / "models").resolve())
    os.makedirs(cache_dir, exist_ok=True)

    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir

    # Initial system memory
    gc.collect()
    time.sleep(2)
    initial_sys = get_system_memory()
    print(format_system_memory(initial_sys, "Initial system memory"))
    print()

    # Load the model
    print("Loading CodeRankEmbed...")
    from sentence_transformers import SentenceTransformer
    import torch

    # Check for best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    before_load = get_system_memory()
    start_time = time.time()

    try:
        model = SentenceTransformer(
            'nomic-ai/CodeRankEmbed',
            device=device,
            cache_folder=cache_dir,
            trust_remote_code=True
        )
        model.eval()

        load_time = time.time() - start_time
        after_load = get_system_memory()

        print(f"\nLoad time: {load_time:.2f} seconds")
        print(format_system_memory(after_load, "After loading model"))

        if before_load and after_load:
            memory_increase = after_load['used_gb'] - before_load['used_gb']
            print(f"\nSystem memory increase: +{memory_increase:.2f} GB")

        # Test with larger batch
        print("\nTesting with larger batch (simulating indexing)...")
        test_texts = [
            f"def function_{i}():\n    return {i} * 2"
            for i in range(100)
        ]

        before_encode = get_system_memory()
        embeddings = model.encode(test_texts, batch_size=32, show_progress_bar=True)
        after_encode = get_system_memory()

        print(f"\nEmbedding shape: {embeddings.shape}")
        if before_encode and after_encode:
            encode_increase = after_encode['used_gb'] - before_encode['used_gb']
            print(f"Memory increase during encoding: +{encode_increase:.2f} GB")

    except Exception as e:
        print(f"Error during model testing: {e}")
        model = None

    # Load another model to simulate model switching if memory allows
    print("\n\nSimulating model switching (loading all-MiniLM-L6-v2)...")
    before_second = get_system_memory()

    try:
        model2 = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=device,
            cache_folder=cache_dir
        )

        after_second = get_system_memory()
        print(format_system_memory(after_second, "After loading second model"))

        if before_second and after_second:
            second_increase = after_second['used_gb'] - before_second['used_gb']
            print(f"Memory increase for second model: +{second_increase:.2f} GB")
            
            if initial_sys:
                total_increase = after_second['used_gb'] - initial_sys['used_gb']
                print(f"\nTotal memory increase (both models): +{total_increase:.2f} GB")
    except Exception as e:
        print(f"Error loading second model: {e}")
        model2 = None

    # Cleanup
    print("\nCleaning up...")
    if 'model' in locals() and model:
        del model
    if 'model2' in locals() and model2:
        del model2
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    time.sleep(3)
    final_sys = get_system_memory()
    print(format_system_memory(final_sys, "After cleanup"))

    if initial_sys and final_sys:
        memory_leaked = final_sys['used_gb'] - initial_sys['used_gb']
        print(f"\nMemory change after cleanup: {memory_leaked:+.2f} GB")