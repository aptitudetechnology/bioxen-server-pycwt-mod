#!/usr/bin/env python3
"""
Phase 2 Validation Test Runner

Runs comprehensive validation tests for the backend system integration.
Organizes tests into categories and provides detailed reporting.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_section(text):
    """Print formatted section header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'-'*70}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'-'*70}{Colors.ENDC}")


def run_test_suite(test_file, test_name, markers=None, verbose=False):
    """Run a test suite and return results."""
    print_section(f"Running: {test_name}")
    
    cmd = ['pytest', test_file, '-v']
    
    if markers:
        cmd.extend(['-m', markers])
    
    if verbose:
        cmd.append('-s')
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(['--cov=pycwt_mod', '--cov-report=term-missing'])
    except ImportError:
        pass
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Parse results
    success = result.returncode == 0
    
    # Print output
    if success:
        print(f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC} ({elapsed:.2f}s)")
    else:
        print(f"{Colors.FAIL}✗ FAILED{Colors.ENDC} ({elapsed:.2f}s)")
        print(f"\n{Colors.WARNING}Output:{Colors.ENDC}")
        print(result.stdout)
        if result.stderr:
            print(f"\n{Colors.FAIL}Errors:{Colors.ENDC}")
            print(result.stderr)
    
    return {
        'name': test_name,
        'success': success,
        'elapsed': elapsed,
        'output': result.stdout,
        'errors': result.stderr
    }


def main():
    """Run all Phase 2 validation tests."""
    print_header("PHASE 2 VALIDATION TEST SUITE")
    
    # Get the project root
    project_root = Path(__file__).parent
    test_dir = project_root / 'src' / 'pycwt_mod' / 'tests'
    
    if not test_dir.exists():
        print(f"{Colors.FAIL}Error: Test directory not found: {test_dir}{Colors.ENDC}")
        return 1
    
    # Change to project root for pytest
    os.chdir(project_root)
    
    print(f"Project root: {project_root}")
    print(f"Test directory: {test_dir}")
    
    results = []
    
    # ========================================================================
    # Category 1: Backend System Tests
    # ========================================================================
    print_header("Category 1: Backend System Tests")
    
    test_suites = [
        (test_dir / 'backends' / 'test_base.py', 
         'Backend Base Class Tests'),
        (test_dir / 'backends' / 'test_registry.py', 
         'Backend Registry Tests'),
        (test_dir / 'backends' / 'test_sequential.py', 
         'Sequential Backend Tests'),
        (test_dir / 'backends' / 'test_joblib.py', 
         'Joblib Backend Tests'),
    ]
    
    for test_file, test_name in test_suites:
        if test_file.exists():
            result = run_test_suite(str(test_file), test_name)
            results.append(result)
        else:
            print(f"{Colors.WARNING}⚠ Skipped: {test_name} (file not found){Colors.ENDC}")
    
    # ========================================================================
    # Category 2: Integration Tests
    # ========================================================================
    print_header("Category 2: Integration Tests")
    
    integration_test = test_dir / 'test_wct_significance_integration.py'
    if integration_test.exists():
        result = run_test_suite(
            str(integration_test),
            'WCT Significance Integration Tests',
            verbose=True
        )
        results.append(result)
    else:
        print(f"{Colors.WARNING}⚠ Skipped: Integration tests (file not found){Colors.ENDC}")
    
    # ========================================================================
    # Category 3: Performance Tests (Optional - marked as slow)
    # ========================================================================
    print_header("Category 3: Performance Tests (Optional)")
    
    run_performance = input(
        f"\n{Colors.OKCYAN}Run performance tests? These may take several minutes. [y/N]: {Colors.ENDC}"
    ).lower().strip()
    
    if run_performance == 'y':
        perf_test = test_dir / 'test_performance.py'
        if perf_test.exists():
            result = run_test_suite(
                str(perf_test),
                'Performance Validation Tests',
                markers='slow',
                verbose=True
            )
            results.append(result)
        else:
            print(f"{Colors.WARNING}⚠ Skipped: Performance tests (file not found){Colors.ENDC}")
    else:
        print(f"{Colors.OKCYAN}Skipping performance tests{Colors.ENDC}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_time = sum(r['elapsed'] for r in results)
    
    print(f"Total test suites: {total_tests}")
    print(f"{Colors.OKGREEN}Passed: {passed_tests}{Colors.ENDC}")
    if failed_tests > 0:
        print(f"{Colors.FAIL}Failed: {failed_tests}{Colors.ENDC}")
    print(f"Total time: {total_time:.2f}s")
    
    # Detailed results
    print(f"\n{Colors.BOLD}Detailed Results:{Colors.ENDC}")
    for i, result in enumerate(results, 1):
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if result['success'] else f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"  {i}. {status} {result['name']} ({result['elapsed']:.2f}s)")
    
    # Failed tests details
    if failed_tests > 0:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Test Details:{Colors.ENDC}")
        for result in results:
            if not result['success']:
                print(f"\n{Colors.FAIL}✗ {result['name']}{Colors.ENDC}")
                print(f"{'='*70}")
                print(result['output'])
                if result['errors']:
                    print(f"\nErrors:")
                    print(result['errors'])
                print(f"{'='*70}")
    
    # Phase 2 completion status
    print_header("PHASE 2 STATUS")
    
    if failed_tests == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✓ PHASE 2 VALIDATION COMPLETE!{Colors.ENDC}")
        print(f"\n{Colors.OKGREEN}All validation tests passed.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Backend system integration is validated and ready.{Colors.ENDC}")
        print(f"\n{Colors.OKCYAN}Next step: Phase 3 - Documentation{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}✗ PHASE 2 VALIDATION INCOMPLETE{Colors.ENDC}")
        print(f"\n{Colors.FAIL}{failed_tests} test suite(s) failed.{Colors.ENDC}")
        print(f"{Colors.WARNING}Please review the failed tests above and fix issues.{Colors.ENDC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
