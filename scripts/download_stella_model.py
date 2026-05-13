#!/usr/bin/env python3
"""
Download Stella ONNX model from Hugging Face using hf CLI.

This script downloads the Stella embedding model ONNX weights and all necessary
configuration files for use with Qdrant RAG MCP server.

Features:
- Download from Hugging Face using hf CLI
- Support for multiple quantization options
- Automatic verification of downloaded files
- Can also copy from local cache if available

Usage:
    python download_stella_model.py [--quantization INT8|FP16|Q4|...] [--output-dir OUTPUT_DIR]
    python download_stella_model.py --local-source /path/to/models

Examples:
    python download_stella_model.py                                    # Download int8 from HF
    python download_stella_model.py --quantization fp16               # Download fp16 version
    python download_stella_model.py --local-source /mnt/Meta/LLM/onnx # Copy from local
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

# Model configuration
MODEL_ID = "NovaSearch/stella_en_400M_v5"
ONNX_COMMIT = "154316358e2c8bb71ef0fe7473fd12123acb293e"

# Quantization options and their corresponding model files
QUANTIZATION_MAP = {
    "int8": "onnx/model_int8.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "q4": "onnx/model_q4.onnx",
    "q4f16": "onnx/model_q4f16.onnx",
    "uint8": "onnx/model_uint8.onnx",
    "bnb4": "onnx/model_bnb4.onnx",
    "quantized": "onnx/model_quantized.onnx",
}

# Config files to download from main branch
CONFIG_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "bpe_vocab_32000.json",
    "README.md",
]

# HF CLI path (system default)
HF_CLI = "hf"


def check_hf_cli():
    """Check if hf CLI is available."""
    try:
        result = subprocess.run(
            [HF_CLI, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "hf CLI not found"


def run_hf_download(repo_id, files, revision=None, local_dir=None, repo_type="model", token=None, max_workers=4):
    """Run hf download command."""
    cmd = [
        HF_CLI, "download", repo_id,
        "--type", repo_type,
        "--max-workers", str(max_workers),
    ]

    if revision:
        cmd.extend(["--revision", revision])

    if local_dir:
        cmd.extend(["--local-dir", local_dir])

    if token:
        cmd.extend(["--token", token])

    # Add files to download
    cmd.extend(files)

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"


def download_onnx_model(output_dir, quantization="int8", token=None, max_workers=4):
    """Download ONNX model from specific commit."""
    print(f"\n{'='*60}")
    print(f"Downloading ONNX Model ({quantization.upper()})")
    print(f"{'='*60}")

    # Get the model file for this quantization
    model_file = QUANTIZATION_MAP.get(quantization, QUANTIZATION_MAP["int8"])

    # Download to temporary location first
    temp_dir = os.path.join(output_dir, "temp_onnx")
    os.makedirs(temp_dir, exist_ok=True)

    # Download the specific ONNX file
    success, message = run_hf_download(
        repo_id=MODEL_ID,
        files=[model_file],
        revision=ONNX_COMMIT,
        local_dir=temp_dir,
        token=token,
        max_workers=max_workers
    )

    if not success:
        print(f"Failed to download ONNX model: {message}")
        return False

    # Move and rename to model.onnx
    source_path = os.path.join(temp_dir, model_file)
    dest_path = os.path.join(output_dir, "model.onnx")

    if os.path.exists(source_path):
        shutil.move(source_path, dest_path)
        print(f"✓ Downloaded and renamed to: {dest_path}")
    else:
        # Try to find the file in the temp directory
        found = False
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".onnx"):
                    os.rename(os.path.join(root, file), dest_path)
                    print(f"✓ Downloaded and renamed to: {dest_path}")
                    found = True
                    break
        if not found:
            print("✗ Could not find ONNX model file")
            return False

    # Cleanup temp directory
    subprocess.run(["rm", "-rf", temp_dir], check=False)

    return True


def download_config_files(output_dir, token=None, max_workers=4):
    """Download configuration files from main branch."""
    print(f"\n{'='*60}")
    print("Downloading Configuration Files")
    print(f"{'='*60}")

    # Download all config files
    files_to_download = []
    for file in CONFIG_FILES:
        files_to_download.append(file)

    success, message = run_hf_download(
        repo_id=MODEL_ID,
        files=files_to_download,
        local_dir=output_dir,
        token=token,
        max_workers=max_workers
    )

    if success:
        print(f"✓ Configuration files downloaded to: {output_dir}")
        return True
    else:
        print(f"Warning: Some config files may not have downloaded: {message}")
        return True  # Still return True as some files are optional


def copy_from_local(local_source, output_dir, quantization="int8"):
    """Copy model files from local source."""
    print(f"\n{'='*60}")
    print(f"Copying from Local Source")
    print(f"{'='*60}")
    print(f"Source: {local_source}")
    print(f"Target: {output_dir}")

    # Check if source exists
    if not os.path.exists(local_source):
        print(f"✗ Local source not found: {local_source}")
        return False

    # Look for the stella model directory
    stella_dirs = []
    for root, dirs, files in os.walk(local_source):
        if "stella_en_400M_v5" in root:
            stella_dirs.append(root)

    if not stella_dirs:
        print(f"✗ stella_en_400M_v5 not found in {local_source}")
        return False

    # Use the first match
    source_base = stella_dirs[0]
    print(f"Found Stella model at: {source_base}")

    # Check for quantization subdirectory
    quant_dir = os.path.join(source_base, quantization)
    if os.path.exists(quant_dir):
        source_base = quant_dir
        print(f"Using quantization subdirectory: {quant_dir}")

    # Copy all files
    for item in os.listdir(source_base):
        src_path = os.path.join(source_base, item)
        dst_path = os.path.join(output_dir, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

        print(f"  Copied: {item}")

    # Check if model.onnx needs to be renamed
    model_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(model_onnx):
        # Look for any .onnx file and rename it
        for item in os.listdir(output_dir):
            if item.endswith(".onnx"):
                os.rename(os.path.join(output_dir, item), model_onnx)
                print(f"  Renamed {item} to model.onnx")
                break

    return True


def verify_download(output_dir):
    """Verify that required files exist."""
    print(f"\n{'='*60}")
    print("Verifying Download")
    print(f"{'='*60}")

    required_files = ["model.onnx"]
    all_present = True

    for file in required_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"✗ {file} - MISSING")
            all_present = False

    # Check for optional config files
    optional_files = ["tokenizer.json", "config.json", "tokenizer_config.json", "modules.json"]
    for file in optional_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"  {file} - Not found (optional)")

    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Download Stella ONNX model from Hugging Face"
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "fp16", "q4", "q4f16", "uint8", "bnb4", "quantized"],
        default="int8",
        help="Quantization type (default: int8)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/models/stella_en_400M_v5/<quantization>)"
    )
    parser.add_argument(
        "--local-source",
        default=None,
        help="Local directory containing pre-downloaded models (e.g., /mnt/Meta/LLM/onnx)"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of download workers (default: 4)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default based on project structure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(
            project_root,
            "data",
            "models",
            "stella_en_400M_v5",
            args.quantization
        )

    print(f"Output directory: {output_dir}")
    print(f"Quantization: {args.quantization}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if local source is provided
    if args.local_source:
        if not copy_from_local(args.local_source, output_dir, args.quantization):
            print("\n✗ Failed to copy from local source")
            sys.exit(1)
    else:
        # Check hf CLI
        available, version = check_hf_cli()
        if not available:
            print(f"Error: hf CLI not found. Please install it first.")
            print("Install from: https://github.com/huggingface/hf-hub")
            sys.exit(1)

        print(f"Using hf CLI: {version}")

        # Download ONNX model
        if not download_onnx_model(output_dir, args.quantization, args.token, args.max_workers):
            print("\n✗ Failed to download model")
            sys.exit(1)

        # Download config files
        download_config_files(output_dir, args.token, args.max_workers)

    # Verify download
    if verify_download(output_dir):
        print(f"\n{'='*60}")
        print("✓ Download completed successfully!")
        print(f"{'='*60}")
        print(f"\nModel location: {output_dir}/model.onnx")
        print(f"\nTo use with Qdrant RAG MCP, update config/server_config.json:")
        print(f'  "embeddings": {{')
        print(f'    "model": "{output_dir}"')
        print(f"  }}")
    else:
        print("\n✗ Download verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
