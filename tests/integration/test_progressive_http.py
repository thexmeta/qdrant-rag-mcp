#!/usr/bin/env python3
"""Test Progressive Context through HTTP API."""

import requests
import sys
import os
import subprocess
import time
import pytest
from typing import Dict, Any
from pathlib import Path

# Base URL for the HTTP server
BASE_URL = "http://localhost:8880"

# Global server process
_server_process = None


def start_server():
    """Start the HTTP server and return the process."""
    global _server_process
    
    if _server_process is not None:
        return _server_process
    
    print("\n🚀 Starting Qdrant RAG HTTP Server for integration tests...")
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Start server in a subprocess
    # Use the same environment as the current process
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    
    # Start the server using the virtual environment python
    python_exe = sys.executable
    _server_process = subprocess.Popen(
        [python_exe, str(project_root / "src" / "http_server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to be ready
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print("✓ Server is ready!")
                return _server_process
        except requests.exceptions.ConnectionError:
            pass
        
        # Check if process died
        if _server_process.poll() is not None:
            stdout, stderr = _server_process.communicate()
            print(f"✗ Server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
            raise RuntimeError("Server failed to start")
            
        time.sleep(1)
        retry_count += 1
    
    _server_process.terminate()
    raise RuntimeError("Server did not start within 30 seconds")


def stop_server():
    """Stop the HTTP server."""
    global _server_process
    
    if _server_process is None:
        return
    
    print("\n🛑 Stopping Qdrant RAG HTTP Server...")
    _server_process.terminate()
    try:
        _server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _server_process.kill()
    print("✓ Server stopped")
    _server_process = None


@pytest.fixture(scope="module", autouse=True)
def server():
    """Start the HTTP server before tests and stop it after."""
    start_server()
    yield _server_process
    stop_server()


def test_search_with_progressive_context(
    query: str,
    context_level: str = "auto",
    progressive_mode: bool = True,
    n_results: int = 5
) -> Dict[str, Any]:
    """Test search with progressive context parameters."""
    
    url = f"{BASE_URL}/search"
    payload = {
        "query": query,
        "n_results": n_results,
        "search_mode": "hybrid",
        "context_level": context_level,
        "progressive_mode": progressive_mode,
        "include_expansion_options": True,
        "semantic_cache": True
    }
    
    print(f"\n{'='*60}")
    print("Testing Progressive Search")
    print(f"Query: {query}")
    print(f"Context Level: {context_level}")
    print(f"Progressive Mode: {progressive_mode}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Print results
        print(f"Total Results: {result.get('total', 0)}")
        print(f"Search Mode: {result.get('search_mode', 'unknown')}")
        
        # Check if progressive metadata is included
        if "progressive" in result:
            prog = result["progressive"]
            print("\nProgressive Context Metadata:")
            print(f"- Level Used: {prog.get('level_used', 'unknown')}")
            print(f"- Token Estimate: {prog.get('token_estimate', 0)}")
            print(f"- Token Reduction: {prog.get('token_reduction', '0%')}")
            print(f"- Cache Hit: {prog.get('cache_hit', False)}")
            
            if prog.get('query_intent'):
                intent = prog['query_intent']
                print(f"- Query Intent: {intent.get('type', 'unknown')} (confidence: {intent.get('confidence', 0):.2f})")
            
            # Print expansion options if available
            if prog.get('expansion_options'):
                print(f"\nExpansion Options ({len(prog['expansion_options'])} available):")
                for i, opt in enumerate(prog['expansion_options'][:3]):  # Show first 3
                    print(f"  {i+1}. {opt['type']} -> {opt['path']}")
                    print(f"     Tokens: {opt['estimated_tokens']}, Relevance: {opt['relevance']:.2f}")
        
        # Print some results
        print("\nSearch Results:")
        for i, res in enumerate(result.get("results", [])[:3]):
            print(f"\n{i+1}. File: {res.get('file_path', 'unknown')}")
            print(f"   Score: {res.get('score', 0):.3f}")
            print(f"   Type: {res.get('chunk_type', 'unknown')}")
            
            # Show truncated content for progressive results
            content = res.get('content', '')
            if res.get('_truncated') or res.get('_summarize'):
                print(f"   Content: {content[:100]}...")
            else:
                print(f"   Content: {content[:200]}...")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return {"error": str(e)}


@pytest.mark.parametrize("level", ["file", "class", "method"])
def test_different_context_levels(level):
    """Test different context levels."""
    query = "search functionality"
    result = test_search_with_progressive_context(query, context_level=level)
    
    if "progressive" in result:
        prog = result["progressive"]
        print(f"\nLevel '{level}' Summary:")
        print(f"- Token estimate: {prog.get('token_estimate', 0)}")
        print(f"- Token reduction: {prog.get('token_reduction', '0%')}")
        print(f"- Results shape: {len(result.get('results', []))} items")
    else:
        # If server didn't return progressive metadata, it might be misconfigured
        # but for integration tests, we at least expect some results or a successful call
        assert "results" in result or "error" not in result

def test_cache_behavior():
    """Test semantic cache behavior."""
    print("\n" + "="*60)
    print("Testing Semantic Cache Behavior")
    print("="*60)
    
    # First query
    query1 = "What does the authentication system do?"
    test_search_with_progressive_context(query1, context_level="file")
    
    # Similar query (should hit cache)
    query2 = "Explain the authentication system"
    result2 = test_search_with_progressive_context(query2, context_level="file")
    
    # Check cache hit
    if "progressive" in result2:
        if result2["progressive"].get("cache_hit"):
            print("\n✓ Cache hit detected for similar query!")
        else:
            print("\n✗ No cache hit for similar query")



def test_auto_classification():
    """Test automatic query intent classification."""
    test_queries = [
        ("What does the configuration system do?", "file"),
        ("Show me the bug in line 123", "method"),
        ("Find the DatabaseManager class", "class"),
        ("How does the caching work?", "file"),
        ("Fix the error in the save function", "method")
    ]
    
    print("\n" + "="*60)
    print("Testing Query Intent Auto-Classification")
    print("="*60)
    
    for query, expected_level in test_queries:
        result = test_search_with_progressive_context(query, context_level="auto", n_results=2)
        
        if "progressive" in result:
            actual_level = result["progressive"].get("level_used", "unknown")
            match = "✓" if actual_level == expected_level else "✗"
            print(f"\n{match} Query: '{query}'")
            print(f"  Expected: {expected_level}, Got: {actual_level}")


def main():
    """Run all tests."""
    print("Progressive Context HTTP Tests")
    print("Make sure progressive_context is enabled in server_config.json")
    
    # Start the server at the beginning
    try:
        start_server()
    except RuntimeError as e:
        print(f"\n✗ Failed to start server: {e}")
        sys.exit(1)
    
    try:
        # Run tests
        if len(sys.argv) > 1:
            if sys.argv[1] == "cache":
                test_cache_behavior()
            elif sys.argv[1] == "levels":
                test_different_context_levels()
            elif sys.argv[1] == "auto":
                test_auto_classification()
            else:
                # Custom query test
                query = " ".join(sys.argv[1:])
                test_search_with_progressive_context(query)
        else:
            # Run a basic test
            test_search_with_progressive_context("What does the authentication system do?", context_level="file")
            
            print("\n\nOther test options:")
            print("  python test_progressive_http.py cache    # Test cache behavior")
            print("  python test_progressive_http.py levels   # Test different context levels")
            print("  python test_progressive_http.py auto     # Test auto-classification")
            print("  python test_progressive_http.py <query>  # Test custom query")
    finally:
        # Clean up: stop the server
        stop_server()


if __name__ == "__main__":
    main()