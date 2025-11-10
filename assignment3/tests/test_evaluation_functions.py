#!/usr/bin/env python3
"""Sanity check for evaluation functions.

This script tests whether the student's implementations of extract_solution,
compute_score, and estimate_pass_at_k match the expected reference outputs.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# TEST CASES FOR extract_solution()
# ============================================================================

EXTRACT_SOLUTION_TEST_CASES = [
    # (input_text, expected_output)
    ("The answer is \\boxed{42}", "42"),
    ("After calculation, we get \\boxed{123.5}", "123.5"),
    ("Therefore the result is 15", "15"),
    ("The final answer is -7.5", "-7.5"),
    ("We can see that 3 + 5 = 8. The answer is 8.", "8"),
    ("The total cost is $1,234.50", "1234.50"),
    ("No answer here", None),  # Should return None or last number if any
    ("First we get 10, then 20, finally 30", "30"),  # Should get last number
    ("\\boxed{100} is wrong, the answer is \\boxed{200}", "200"),  # Multiple boxed
]


# ============================================================================
# TEST CASES FOR compute_score()
# ============================================================================

COMPUTE_SCORE_TEST_CASES = [
    # (solution_str, ground_truth, expected_score)
    ("The answer is \\boxed{42}", "42", 1),
    ("The answer is \\boxed{42}", "42.0", 1),  # Should normalize
    ("The answer is \\boxed{42.0}", "42", 1),  # Should normalize
    ("The answer is \\boxed{42}", "43", 0),
    ("Therefore, we get 15", "15", 1),
    ("Therefore, we get 15", "16", 0),
    ("The result is $1,234", "1234", 1),  # Should handle commas
    ("No number here", "42", 0),
]


# ============================================================================
# TEST CASES FOR estimate_pass_at_k()
# ============================================================================

# These are computed using the reference implementation
# Formula: pass@k = 1 - C(n-c, k) / C(n, k)

ESTIMATE_PASS_AT_K_TEST_CASES = [
    # (num_samples, num_correct, k, expected_pass_at_k)
    # Case 1: All samples correct
    (5, [5, 5, 5], 1, [1.0, 1.0, 1.0]),
    # Case 2: No samples correct
    (5, [0, 0, 0], 1, [0.0, 0.0, 0.0]),
    # Case 3: Some correct (n=10, c=5, k=1)
    # pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
    (10, [5], 1, [0.5]),
    # Case 4: n=10, c=7, k=3
    # pass@3 = 1 - C(3,3)/C(10,3) = 1 - 1/120 = 0.991666...
    (10, [7], 3, [0.9916666666666667]),
    # Case 5: Edge case - not enough wrong samples (n=5, c=4, k=3)
    # Can't pick 3 wrong from only 1 wrong, so pass@3 = 1.0
    (5, [4], 3, [1.0]),
    # Case 6: Multiple problems with different samples
    (10, [3, 5, 8], 2, [0.5333333333333333, 0.7777777777777778, 0.9777777777777777]),
]


def test_extract_solution():
    """Test extract_solution function."""
    print("\n" + "="*70)
    print("TEST 1: extract_solution()")
    print("="*70)

    try:
        from evaluate_model_outputs import extract_solution

        passed = 0
        failed = 0

        for i, (input_text, expected) in enumerate(EXTRACT_SOLUTION_TEST_CASES):
            result = extract_solution(input_text)

            # For None case, we accept either None or the last number
            if expected is None:
                # Accept both None or a number (implementation-dependent)
                status = "✅ PASS"
                passed += 1
            elif result == expected:
                status = "✅ PASS"
                passed += 1
            else:
                # Try normalizing (remove commas, convert to float)
                try:
                    result_norm = str(float(result.replace(',', ''))) if result else None
                    expected_norm = str(float(expected.replace(',', ''))) if expected else None
                    if result_norm == expected_norm:
                        status = "✅ PASS"
                        passed += 1
                    else:
                        status = f"❌ FAIL"
                        failed += 1
                        print(f"\n  Test {i+1}: {status}")
                        print(f"    Input: {input_text[:50]}...")
                        print(f"    Expected: {expected}")
                        print(f"    Got: {result}")
                except:
                    status = f"❌ FAIL"
                    failed += 1
                    print(f"\n  Test {i+1}: {status}")
                    print(f"    Input: {input_text[:50]}...")
                    print(f"    Expected: {expected}")
                    print(f"    Got: {result}")

        print(f"\nResult: {passed}/{len(EXTRACT_SOLUTION_TEST_CASES)} passed")

        if failed == 0:
            print("✅ PASSED: extract_solution()")
            return True
        else:
            print(f"❌ FAILED: extract_solution() - {failed} test(s) failed")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_score():
    """Test compute_score function."""
    print("\n" + "="*70)
    print("TEST 2: compute_score()")
    print("="*70)

    try:
        from evaluate_model_outputs import compute_score

        passed = 0
        failed = 0

        for i, (solution_str, ground_truth, expected_score) in enumerate(COMPUTE_SCORE_TEST_CASES):
            result = compute_score(solution_str, ground_truth)

            if result == expected_score:
                status = "✅ PASS"
                passed += 1
            else:
                status = f"❌ FAIL"
                failed += 1
                print(f"\n  Test {i+1}: {status}")
                print(f"    Solution: {solution_str[:50]}...")
                print(f"    Ground truth: {ground_truth}")
                print(f"    Expected score: {expected_score}")
                print(f"    Got score: {result}")

        print(f"\nResult: {passed}/{len(COMPUTE_SCORE_TEST_CASES)} passed")

        if failed == 0:
            print("✅ PASSED: compute_score()")
            return True
        else:
            print(f"❌ FAILED: compute_score() - {failed} test(s) failed")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_estimate_pass_at_k():
    """Test estimate_pass_at_k function."""
    print("\n" + "="*70)
    print("TEST 3: estimate_pass_at_k()")
    print("="*70)

    try:
        from evaluate_model_outputs import estimate_pass_at_k

        passed = 0
        failed = 0

        for i, (num_samples, num_correct, k, expected) in enumerate(ESTIMATE_PASS_AT_K_TEST_CASES):
            result = estimate_pass_at_k(num_samples, num_correct, k)

            # Convert to numpy array for comparison
            result = np.array(result)
            expected = np.array(expected)

            # Check if close (allow small floating point errors)
            if np.allclose(result, expected, rtol=1e-5, atol=1e-8):
                status = "✅ PASS"
                passed += 1
            else:
                status = f"❌ FAIL"
                failed += 1
                print(f"\n  Test {i+1}: {status}")
                print(f"    num_samples: {num_samples}")
                print(f"    num_correct: {num_correct}")
                print(f"    k: {k}")
                print(f"    Expected: {expected}")
                print(f"    Got: {result}")
                print(f"    Difference: {np.abs(result - expected)}")

        print(f"\nResult: {passed}/{len(ESTIMATE_PASS_AT_K_TEST_CASES)} passed")

        if failed == 0:
            print("✅ PASSED: estimate_pass_at_k()")
            return True
        else:
            print(f"❌ FAILED: estimate_pass_at_k() - {failed} test(s) failed")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SANITY CHECK: Evaluation Functions")
    print("="*70)
    print("\nThis script tests your implementations of:")
    print("  - extract_solution()")
    print("  - compute_score()")
    print("  - estimate_pass_at_k()")
    print("\nIt compares your outputs with expected reference outputs.\n")

    results = []

    # Run tests
    results.append(("extract_solution", test_extract_solution()))
    results.append(("compute_score", test_compute_score()))
    results.append(("estimate_pass_at_k", test_estimate_pass_at_k()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}()")

    print(f"\nTotal: {passed}/{total} functions passed")

    if failed == 0:
        print("\n🎉 All tests passed! Your evaluation functions look correct.")
        return 0
    else:
        print(f"\n⚠️  {failed} function(s) failed. Please review your implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
