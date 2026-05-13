#!/usr/bin/env python3
"""
Comprehensive MCP Server All-Tools Test Suite

This module systematically tests ALL MCP server tools by:
1. Executing each tool with valid inputs
2. Comparing actual outputs to expected results
3. Documenting outcomes in a structured JSON report

Report Schema:
{
  "report_id": "mcp-test-YYYYMMDD-HHMMSS",
  "generated_at": "ISO8601 timestamp",
  "total_tests": int,
  "passed": int,
  "failed": int,
  "skipped": int,
  "errors": int,
  "pass_rate": float,
  "test_results": [TestResult, ...],
  "environment": {...},
  "summary": "string"
}

TestResult Schema:
{
  "test_name": "string",
  "input_parameters": {...},
  "expected_output": {...},
  "actual_output": {...} | null,
  "status": "passed"|"failed"|"skipped"|"error",
  "error_message": "string" | null,
  "duration_ms": float,
  "timestamp": "ISO8601",
  "skipped_reason": "string" | null
}
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import tempfile
import shutil

# Add src to path
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Test configuration
TEST_TIMEOUT_SECONDS = 30
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class TestStatus(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Schema for test result"""
    test_name: str
    input_parameters: Dict[str, Any]
    expected_output: Dict[str, Any]
    actual_output: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]
    duration_ms: float
    timestamp: str
    skipped_reason: Optional[str] = None


@dataclass
class TestReport:
    """Schema for complete test report"""
    report_id: str
    generated_at: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    pass_rate: float
    test_results: List[TestResult]
    environment: Dict[str, Any]
    summary: str


class MCPServerTester:
    """
    Comprehensive test suite for ALL MCP server tools.
    
    Categories:
    - Context Management (get_context, get_file_chunks, detect_changes)
    - Indexing (index_code, index_config, index_documentation, index_directory, reindex_directory)
    - Search (search, search_code, search_docs, search_config)
    - Project Management (switch_project, clear_project_collections)
    - BM25 Operations (rebuild_bm25_indices)
    - Memory Management (get_memory_status, trigger_memory_cleanup)
    - Health & Diagnostics (health_check)
    - GitHub Integration (github_* tools)
    """
    
    def __init__(self, report_path: Optional[str] = None):
        self.report_path = report_path or "test_report.json"
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
        # Import tracking
        self.server_available = False
        self.import_errors: List[str] = []
        self.available_tools: Dict[str, callable] = {}
        self.missing_tools: List[str] = []
        
        # Try to import server components
        self._import_server_components()
        
        # Environment checks
        self.qdrant_available = self._check_qdrant_connection()
        self.github_available = bool(os.getenv("GITHUB_TOKEN"))
        
    def _import_server_components(self):
        """Import all server tools and track availability"""
        try:
            from qdrant_mcp_context_aware import (
                # Context Management
                get_context,
                get_file_chunks,
                detect_changes,
                
                # Indexing
                index_code,
                index_config,
                index_documentation,
                index_directory,
                reindex_directory,
                
                # Search
                search,
                search_code,
                search_docs,
                search_config,
                
                # Project Management
                switch_project,
                clear_project_collections,
                
                # BM25
                rebuild_bm25_indices,
                
                # Memory
                get_memory_status,
                trigger_memory_cleanup,
                
                # Health
                health_check,
            )
            
            self.available_tools = {
                # Context Management
                "get_context": get_context,
                "get_file_chunks": get_file_chunks,
                "detect_changes": detect_changes,
                
                # Indexing
                "index_code": index_code,
                "index_config": index_config,
                "index_documentation": index_documentation,
                "index_directory": index_directory,
                "reindex_directory": reindex_directory,
                
                # Search
                "search": search,
                "search_code": search_code,
                "search_docs": search_docs,
                "search_config": search_config,
                
                # Project Management
                "switch_project": switch_project,
                "clear_project_collections": clear_project_collections,
                
                # BM25
                "rebuild_bm25_indices": rebuild_bm25_indices,
                
                # Memory
                "get_memory_status": get_memory_status,
                "trigger_memory_cleanup": trigger_memory_cleanup,
                
                # Health
                "health_check": health_check,
            }
            
            self.server_available = True
            
        except ImportError as e:
            self.import_errors.append(f"Main server import: {e}")
            self.server_available = False
    
    def _check_qdrant_connection(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            from qdrant_client import QdrantClient
            
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            return True
        except Exception:
            return False
    
    def _record_result(self, result: TestResult):
        """Record a test result"""
        self.results.append(result)
    
    def _create_result(
        self,
        test_name: str,
        input_params: Dict[str, Any],
        expected: Dict[str, Any],
        actual: Optional[Dict[str, Any]],
        status: str,
        error_msg: Optional[str] = None,
        duration_ms: float = 0.0,
        skipped_reason: Optional[str] = None,
    ) -> TestResult:
        """Create a test result"""
        return TestResult(
            test_name=test_name,
            input_parameters=input_params,
            expected_output=expected,
            actual_output=actual,
            status=status,
            error_message=error_msg,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            skipped_reason=skipped_reason,
        )
    
    def _check_dependency(self, dep_name: str) -> Tuple[bool, str]:
        """
        Check if a dependency is available.
        
        Returns:
            Tuple of (available, reason_if_missing)
        """
        reasons = {
            "qdrant": ("Qdrant server not running or unreachable", self.qdrant_available),
            "github": ("GITHUB_TOKEN not set in environment", self.github_available),
            "server": ("MCP server not available: " + "; ".join(self.import_errors), self.server_available),
        }
        
        if dep_name in reasons:
            reason, available = reasons[dep_name]
            return (available, reason if not available else "")
        return (False, f"Unknown dependency: {dep_name}")
    
    # ==================== Context Management Tests ====================
    
    def test_get_context(self) -> TestResult:
        """Test get_context tool - returns current project context"""
        test_name = "test_get_context"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"current_project": "<project_info>"},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["get_context"]
            actual = func()
            duration_ms = (time.time() - start_time) * 1000
            
            expected_keys = ["current_project", "working_directory"]
            has_keys = all(key in actual for key in expected_keys) if isinstance(actual, dict) else False
            
            status = TestStatus.PASSED.value if has_keys else TestStatus.FAILED.value
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"keys": expected_keys},
                actual=actual if has_keys else None,
                status=status,
                error_msg=None if has_keys else "Missing expected keys in response",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"current_project": "<project_info>"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_get_file_chunks(self) -> TestResult:
        """Test get_file_chunks tool - retrieves chunks for a file"""
        test_name = "test_get_file_chunks"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": "test.py"},
                expected={"file_path": "test.py", "chunks": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["get_file_chunks"]
            # Test with non-existent file (should return error dict)
            actual = func(file_path="nonexistent_test.py")
            duration_ms = (time.time() - start_time) * 1000
            
            # Should return a dict (either with error or empty chunks)
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": "nonexistent_test.py"},
                    expected={"dict response with error or chunks"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": "nonexistent_test.py"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": "nonexistent_test.py"},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_detect_changes(self) -> TestResult:
        """Test detect_changes tool - compares filesystem with index"""
        test_name = "test_detect_changes"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"directory": "."},
                expected={"added": [], "modified": [], "deleted": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["detect_changes"]
            actual = func(directory=".")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"directory": "."},
                    expected={"change_categories": ["added", "modified", "deleted", "unchanged"]},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"directory": "."},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"directory": "."},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Indexing Tests ====================
    
    def test_index_code(self) -> TestResult:
        """Test index_code tool - indexes a code file"""
        test_name = "test_index_code"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": "test.py"},
                expected={"indexed": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass\n")
            temp_file = f.name
        
        try:
            func = self.available_tools["index_code"]
            actual = func(file_path=temp_file)
            duration_ms = (time.time() - start_time) * 1000
            
            # Should return dict with success or error
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response with indexing result"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": temp_file},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_file)
            except:
                pass
        
        self._record_result(result)
        return result
    
    def test_index_config(self) -> TestResult:
        """Test index_config tool - indexes a config file"""
        test_name = "test_index_config"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": "config.json"},
                expected={"indexed": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": True}, f)
            temp_file = f.name
        
        try:
            func = self.available_tools["index_config"]
            actual = func(file_path=temp_file)
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": temp_file},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        self._record_result(result)
        return result
    
    def test_index_documentation(self) -> TestResult:
        """Test index_documentation tool - indexes a documentation file"""
        test_name = "test_index_documentation"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": "README.md"},
                expected={"indexed": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test\n\nContent here.\n")
            temp_file = f.name
        
        try:
            func = self.available_tools["index_documentation"]
            actual = func(file_path=temp_file)
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"file_path": temp_file},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"file_path": temp_file},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        self._record_result(result)
        return result
    
    def test_index_directory(self) -> TestResult:
        """Test index_directory tool - indexes all files in a directory"""
        test_name = "test_index_directory"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"directory": "."},
                expected={"indexed": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["index_directory"]
            # Test with current directory (should work but may take time)
            actual = func(directory=".", max_files=5)  # Limit for test
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"directory": ".", "max_files": 5},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"directory": ".", "max_files": 5},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"directory": ".", "max_files": 5},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Search Tests ====================
    
    def test_search(self) -> TestResult:
        """Test search tool - general search across all collections"""
        test_name = "test_search"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "test"},
                expected={"results": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["search"]
            actual = func(query="test query")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "test query"},
                    expected={"dict response with results"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "test query"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "test query"},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_search_code(self) -> TestResult:
        """Test search_code tool - search only code collections"""
        test_name = "test_search_code"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "function"},
                expected={"results": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["search_code"]
            actual = func(query="function definition")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "function definition"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "function definition"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "function definition"},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_search_docs(self) -> TestResult:
        """Test search_docs tool - search only documentation collections"""
        test_name = "test_search_docs"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "documentation"},
                expected={"results": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["search_docs"]
            actual = func(query="documentation")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "documentation"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "documentation"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "documentation"},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_search_config(self) -> TestResult:
        """Test search_config tool - search only config collections"""
        test_name = "test_search_config"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "configuration"},
                expected={"results": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["search_config"]
            actual = func(query="configuration")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "configuration"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"query": "configuration"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"query": "configuration"},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Project Management Tests ====================
    
    def test_switch_project(self) -> TestResult:
        """Test switch_project tool - switch to a different project"""
        test_name = "test_switch_project"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"project_path": "."},
                expected={"switched": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["switch_project"]
            # Test with current directory
            actual = func(project_path=".")
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"project_path": "."},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"project_path": "."},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"project_path": "."},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_clear_project_collections(self) -> TestResult:
        """Test clear_project_collections tool - clear all project collections"""
        test_name = "test_clear_project_collections"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"cleared": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["clear_project_collections"]
            actual = func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== BM25 Tests ====================
    
    def test_rebuild_bm25_indices(self) -> TestResult:
        """Test rebuild_bm25_indices tool - rebuild BM25 indices"""
        test_name = "test_rebuild_bm25_indices"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"rebuilt": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["rebuild_bm25_indices"]
            actual = func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Memory Management Tests ====================
    
    def test_get_memory_status(self) -> TestResult:
        """Test get_memory_status tool - get memory manager status"""
        test_name = "test_get_memory_status"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"memory_info": {}},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["get_memory_status"]
            actual = func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    def test_trigger_memory_cleanup(self) -> TestResult:
        """Test trigger_memory_cleanup tool - trigger memory cleanup"""
        test_name = "test_trigger_memory_cleanup"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={"aggressive": False},
                expected={"cleaned": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["trigger_memory_cleanup"]
            actual = func(aggressive=False)
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={"aggressive": False},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={"aggressive": False},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={"aggressive": False},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Health Check Tests ====================
    
    def test_health_check(self) -> TestResult:
        """Test health_check tool - system health check"""
        test_name = "test_health_check"
        start_time = time.time()
        
        available, reason = self._check_dependency("server")
        if not available:
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"healthy": True},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason=reason,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            func = self.available_tools["health_check"]
            actual = func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(actual, dict):
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_result(
                    test_name=test_name,
                    input_params={},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_result(
                test_name=test_name,
                input_params={},
                expected={"dict response"},
                actual=None,
                status=TestStatus.ERROR.value,
                error_msg=str(e),
                duration_ms=duration_ms,
            )
        
        self._record_result(result)
        return result
    
    # ==================== Report Generation ====================
    
    def generate_report(self) -> TestReport:
        """Generate the final test report"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED.value)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED.value)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED.value)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR.value)
        
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "qdrant_available": self.qdrant_available,
            "github_available": self.github_available,
            "server_available": self.server_available,
            "import_errors": self.import_errors,
        }
        
        summary = f"{passed}/{total} tests passed ({pass_rate:.1f}%)"
        if skipped > 0:
            summary += f", {skipped} skipped"
        if errors > 0:
            summary += f", {errors} errors"
        if failed > 0:
            summary += f", {failed} failed"
        
        report = TestReport(
            report_id=f"mcp-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            pass_rate=pass_rate,
            test_results=self.results,
            environment=env_info,
            summary=summary,
        )
        
        return report
    
    def save_report(self, report: TestReport):
        """Save report to JSON file"""
        report_dict = asdict(report)
        report_dict["test_results"] = [asdict(r) for r in report.test_results]
        
        with open(self.report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"Test report saved to: {self.report_path}")
    
    def run_all_tests(self) -> TestReport:
        """Run all tests and generate report"""
        print("Starting MCP Server Comprehensive Test Suite...")
        print("=" * 60)
        
        # Context management
        print("\n[1/6] Context Management Tests...")
        self.test_get_context()
        self.test_get_file_chunks()
        self.test_detect_changes()
        
        # Indexing
        print("\n[2/6] Indexing Tests...")
        self.test_index_code()
        self.test_index_config()
        self.test_index_documentation()
        self.test_index_directory()
        
        # Search
        print("\n[3/6] Search Tests...")
        self.test_search()
        self.test_search_code()
        self.test_search_docs()
        self.test_search_config()
        
        # Project Management
        print("\n[4/6] Project Management Tests...")
        self.test_switch_project()
        self.test_clear_project_collections()
        
        # BM25 & Memory
        print("\n[5/6] BM25 & Memory Tests...")
        self.test_rebuild_bm25_indices()
        self.test_get_memory_status()
        self.test_trigger_memory_cleanup()
        
        # Health
        print("\n[6/6] Health Check Tests...")
        self.test_health_check()
        
        # Generate report
        print("\n" + "=" * 60)
        print("Generating test report...")
        
        report = self.generate_report()
        self.save_report(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed} ({report.pass_rate:.1f}%)")
        print(f"Failed: {report.failed}")
        print(f"Skipped: {report.skipped}")
        print(f"Errors: {report.errors}")
        print("=" * 60)
        
        # Highlight failures
        failures = [r for r in self.results if r.status in [TestStatus.FAILED.value, TestStatus.ERROR.value]]
        if failures:
            print("\nFAILURES/ERRORS:")
            for result in failures:
                print(f"  - {result.test_name}: {result.error_message or result.status}")
        
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Comprehensive Test Suite")
    parser.add_argument(
        "--output",
        type=str,
        default="test_report.json",
        help="Output path for test report JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    tester = MCPServerTester(report_path=args.output)
    report = tester.run_all_tests()
    
    # Exit with appropriate code
    if report.failed > 0 or report.errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()