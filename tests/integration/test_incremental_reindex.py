#!/usr/bin/env python3
"""
Test script for incremental reindex functionality.

This script demonstrates the enhanced reindex_directory function
with incremental mode support.
"""

import sys
import os
from pathlib import Path

import sys
import os
from pathlib import Path
import pytest

# The conftest.py already adds src to path, but we can keep it for robustness if needed
# though pytest handles it.

def test_incremental_reindex(temp_test_dir, sample_python_file, sample_config_file, sample_markdown_file):
    """Test the incremental reindex functionality using temporary files"""
    from qdrant_mcp_context_aware import reindex_directory, detect_changes
    
    # Create a project marker so get_current_project works
    (temp_test_dir / ".git").mkdir()
    
    print(f"Testing incremental reindex functionality in {temp_test_dir}...")
    
    # 1. Initialize with a full reindex (using force=True)
    print("\n1. Initializing with full reindex:")
    dir_path = str(temp_test_dir.resolve())
    result = reindex_directory(directory=dir_path, force=True)
    assert result.get('mode') == 'full_reindex'
    
    # 2. Testing incremental reindex (no changes yet)
    print("\n2. Testing incremental reindex (no changes):")
    result = reindex_directory(directory=dir_path, incremental=True, force=False)
    print(f"   Mode: {result.get('mode', 'unknown')}")
    assert result.get('mode') == 'incremental_reindex'
    if 'changes_detected' in result:
        changes = result['changes_detected']
        print(f"   Changes detected: {changes['added']} added, {changes['modified']} modified, {changes['deleted']} deleted, {changes['unchanged']} unchanged")
        assert changes['unchanged'] >= 3
    
    # 3. Modify a file and test incremental again
    print("\n3. Testing incremental after modification:")
    with open(sample_python_file, "a") as f:
        f.write("\n# Added a comment\n")
    
    result = reindex_directory(directory=dir_path, incremental=True, force=False)
    print(f"   Mode: {result.get('mode', 'unknown')}")
    assert result.get('mode') == 'incremental_reindex'
    if 'changes_detected' in result:
        changes = result['changes_detected']
        print(f"   Changes detected: {changes['added']} added, {changes['modified']} modified, {changes['deleted']} deleted, {changes['unchanged']} unchanged")
        assert changes['modified'] >= 1
    
    # 4. Testing detect_changes function separately
    print("\n4. Testing detect_changes function:")
    changes = detect_changes(dir_path)
    assert 'error' not in changes
    
    print("\nIncremental reindex test completed!")


if __name__ == "__main__":
    # If run directly, we might need to mock the fixtures or just run with pytest
    print("Please run this test using pytest: pytest tests/integration/test_incremental_reindex.py")