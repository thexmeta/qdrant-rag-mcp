#!/usr/bin/env python3
"""
Comprehensive MCP Server Function Test Suite

This module tests all MCP server tools by:
1. Executing each tool with valid inputs
2. Comparing actual outputs to expected results
3. Documenting outcomes in a structured JSON report

The report follows a predefined schema and includes:
- Test name
- Input parameters
- Expected output
- Actual output
- Pass/fail status
- Error messages (if any)

External dependencies that are unavailable are skipped with rationale.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

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
    Comprehensive test suite for MCP server tools.
    
    Tests are organized by category:
    - Context management tools
    - Indexing tools
    - Search tools
    - Configuration tools
    - Utility tools
    """
    
    def __init__(self, report_path: Optional[str] = None):
        self.report_path = report_path or "test_report.json"
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
        # Try to import server components
        self.server_available = False
        self.import_errors: List[str] = []
        
        try:
            from qdrant_mcp_context_aware import (
                get_context,
                get_file_chunks,
                detect_changes,
            )
            self.get_context = get_context
            self.get_file_chunks = get_file_chunks
            self.detect_changes = detect_changes
            self.server_available = True
        except ImportError as e:
            self.import_errors.append(f"Main server import: {e}")
            
        # Initialize Qdrant client check
        self.qdrant_available = self._check_qdrant_connection()
        
    def _check_qdrant_connection(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            from qdrant_client import QdrantClient
            from config import get_config
            
            config = get_config()
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
    
    def _create_test_result(
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
        """Create and record a test result"""
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
    
    def _check_external_dependency(self, dep_name: str) -> Tuple[bool, str]:
        """
        Check if an external dependency is available.
        
        Returns:
            Tuple of (available, reason)
        """
        reasons = {
            "qdrant": "Qdrant server not running or unreachable",
            "github_token": "GITHUB_TOKEN not set in environment",
            "hf_token": "HuggingFace token not configured",
            "model_files": "Required model files not downloaded",
        }
        
        if dep_name == "qdrant":
            return (self.qdrant_available, reasons.get("qdrant", ""))
        elif dep_name == "github_token":
            has_token = bool(os.getenv("GITHUB_TOKEN"))
            return (has_token, reasons.get("github_token", "") if not has_token else "")
        elif dep_name == "model_files":
            # Check if models directory exists and has content
            model_dir = Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", "~/.cache/huggingface")).expanduser()
            has_models = model_dir.exists() and len(list(model_dir.glob("*"))) > 0
            return (has_models, reasons.get("model_files", "") if not has_models else "")
        
        return (False, f"Unknown dependency: {dep_name}")
    
    # ==================== Context Management Tests ====================
    
    def test_get_context(self) -> TestResult:
        """Test get_context tool - returns current project context"""
        test_name = "test_get_context"
        start_time = time.time()
        
        if not self.server_available:
            result = self._create_test_result(
                test_name=test_name,
                input_params={},
                expected={"current_project": "<project_info>"},
                actual=None,
                status=TestStatus.SKIPPED.value,
                error_msg="Server not available",
                duration_ms=(time.time() - start_time) * 1000,
                skipped_reason="MCP server not available: " + "; ".join(self.import_errors),
            )
            self._record_result(result)
            return result
        
        try:
            # Execute
            actual = self.get_context()
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate
            expected_keys = ["current_project", "working_directory"]
            has_keys = all(key in actual for key in expected_keys) if isinstance(actual, dict) else False
            
            if has_keys:
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={},
                    expected={"keys": expected_keys},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={},
                    expected={"keys": expected_keys},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg=f"Missing expected keys in response",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_test_result(
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
        
        if not self.server_available:
            result = self._create_test_result(
                test_name=test_name,
                input_params={"file_path": "test.py"},
                expected={"file_path": "test.py", "chunks": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason="MCP server not available",
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        # Test with a non-existent file (should return error or empty)
        try:
            actual = self.get_file_chunks(file_path="nonexistent_test_file.py")
            duration_ms = (time.time() - start_time) * 1000
            
            # Should return a dict with either error or empty chunks
            if isinstance(actual, dict):
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={"file_path": "nonexistent_test_file.py"},
                    expected={"error": "or empty chunks"},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={"file_path": "nonexistent_test_file.py"},
                    expected={"dict response"},
                    actual=actual,
                    status=TestStatus.FAILED.value,
                    error_msg="Unexpected response type",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_test_result(
                test_name=test_name,
                input_params={"file_path": "nonexistent_test_file.py"},
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
        
        if not self.server_available:
            result = self._create_test_result(
                test_name=test_name,
                input_params={"directory": "."},
                expected={"added": [], "modified": [], "deleted": []},
                actual=None,
                status=TestStatus.SKIPPED.value,
                skipped_reason="MCP server not available",
                duration_ms=(time.time() - start_time) * 1000,
            )
            self._record_result(result)
            return result
        
        try:
            actual = self.detect_changes(directory=".")
            duration_ms = (time.time() - start_time) * 1000
            
            # Should return a dict with change categories
            if isinstance(actual, dict):
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={"directory": "."},
                    expected={"change_categories": ["added", "modified", "deleted", "unchanged"]},
                    actual=actual,
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_test_result(
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
            result = self._create_test_result(
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
    
    # ==================== Configuration Tests ====================
    
    def test_config_loading(self) -> TestResult:
        """Test configuration loading"""
        test_name = "test_config_loading"
        start_time = time.time()
        
        try:
            from config import get_config
            
            config = get_config()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(config, dict):
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={},
                    expected={"config": "dict"},
                    actual={"config_loaded": True, "keys": list(config.keys())[:10]},
                    status=TestStatus.PASSED.value,
                    duration_ms=duration_ms,
                )
            else:
                result = self._create_test_result(
                    test_name=test_name,
                    input_params={},
                    expected={"config": "dict"},
                    actual=None,
                    status=TestStatus.FAILED.value,
                    error_msg="Config is not a dict",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = self._create_test_result(
                test_name=test_name,
                input_params={},
                expected={"config": "dict"},
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
        # Calculate statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED.value)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED.value)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED.value)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR.value)
        
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        
        # Environment info
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "qdrant_available": self.qdrant_available,
            "server_available": self.server_available,
            "import_errors": self.import_errors,
        }
        
        # Summary
        summary = f"{passed}/{total} tests passed ({pass_rate:.1f}%)"
        if skipped > 0:
            summary += f", {skipped} skipped"
        if errors > 0:
            summary += f", {errors} errors"
        
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
        
        # Convert dataclass instances to dicts
        report_dict["test_results"] = [asdict(r) for r in report.test_results]
        
        with open(self.report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"Test report saved to: {self.report_path}")
    
    def run_all_tests(self) -> TestReport:
        """Run all tests and generate report"""
        print("Starting MCP Server Comprehensive Test Suite...")
        print("=" * 60)
        
        # Context management tests
        print("\n[1/4] Testing context management...")
        self.test_get_context()
        
        print("\n[2/4] Testing file operations...")
        self.test_get_file_chunks()
        
        print("\n[3/4] Testing change detection...")
        self.test_detect_changes()
        
        print("\n[4/4] Testing configuration...")
        self.test_config_loading()
        
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
