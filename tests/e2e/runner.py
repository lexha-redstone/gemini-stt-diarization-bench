"""E2E Test Runner and Execution Reporter.

Runs all 4 tiers of E2E tests using pytest and prints a comprehensive execution summary.
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PYTHON_BIN = sys.executable


def run_e2e_tests():
    print("=" * 70)
    print("RUNNING 4-TIER E2E TEST SUITE")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python: {PYTHON_BIN}")
    print("=" * 70)

    cmd = [
        PYTHON_BIN,
        "-m",
        "pytest",
        os.path.join(PROJECT_ROOT, "tests/e2e"),
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("ALL E2E TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"E2E TESTS FAILED WITH EXIT CODE {result.returncode}")
        print("=" * 70)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_e2e_tests())
