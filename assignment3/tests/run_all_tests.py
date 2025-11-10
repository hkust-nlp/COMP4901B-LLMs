#!/usr/bin/env python3
"""Master test runner for all homework sanity checks.

This script runs all sanity check tests and provides a comprehensive report.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test_module(module_name, description):
    """Run a test module and return the result."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)

    try:
        # Import the module dynamically
        import importlib.util

        test_file = Path(__file__).parent / f"{module_name}.py"
        if not test_file.exists():
            print(f"❌ ERROR: Test file not found: {test_file}")
            return False

        spec = importlib.util.spec_from_file_location(module_name, test_file)
        if spec is None or spec.loader is None:
            print(f"❌ ERROR: Could not load module spec for {module_name}")
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'main'):
            print(f"❌ ERROR: Module {module_name} does not have a main() function")
            return False

        result = module.main()
        return result == 0  # 0 means success

    except Exception as e:
        print(f"❌ ERROR running {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test modules."""
    print("\n" + "="*80)
    print("HOMEWORK SANITY CHECK - MASTER TEST RUNNER")
    print("="*80)
    print("\nThis script runs all sanity checks for the GSM8K homework.")
    print("It will test your implementations of:")
    print("  1. format_prompts() in inference_vllm.py")
    print("  2. extract_solution() in evaluate_model_outputs.py")
    print("  3. compute_score() in evaluate_model_outputs.py")
    print("  4. estimate_pass_at_k() in evaluate_model_outputs.py")
    print("\n" + "="*80)

    # Test modules to run
    tests = [
        ("test_format_prompts", "Prompt Formatting Tests (inference_vllm.py)"),
        ("test_evaluation_functions", "Evaluation Functions Tests (evaluate_model_outputs.py)"),
    ]

    results = []

    # Run each test module
    for module_name, description in tests:
        result = run_test_module(module_name, description)
        results.append((description, result))

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    errors = sum(1 for _, result in results if result is None)
    total = len(results)

    for description, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  ERROR"
        print(f"{status}: {description}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    # Return code
    if failed == 0 and errors == 0:
        print("\n" + "="*80)
        print("🎉 CONGRATULATIONS! All sanity checks passed!")
        print("="*80)
        print("\nYour implementations appear to be correct.")
        print("Next steps:")
        print("  1. Test with real inference (if you haven't already)")
        print("  2. Make sure your code runs on the full GSM8K dataset")
        print("  3. Proceed to the training part if applicable")
        print("\n" + "="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print(f"\n{failed} test suite(s) failed, {errors} error(s) occurred.")
        print("\nPlease review the detailed output above and fix the failing tests.")
        print("You can run individual test files to debug:")
        print("  python tests/test_format_prompts.py")
        print("  python tests/test_evaluation_functions.py")
        print("\n" + "="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
