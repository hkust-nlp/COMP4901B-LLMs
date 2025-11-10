# Homework Sanity Check Tests

This directory contains automated tests to verify your homework implementations.

## Overview

The tests check the correctness of your implementations by comparing outputs with reference implementations.

## Test Files

### 1. `test_format_prompts.py`
Tests the `format_prompts()` function in `inference_vllm.py`.

**What it tests:**
- Zero-shot prompt formatting (without chat template)
- Few-shot prompt formatting (with 8 examples)
- Chat template application (with real tokenizer)
- System message inclusion

**Run individually:**
```bash
python tests/test_format_prompts.py
```

### 2. `test_evaluation_functions.py`
Tests the evaluation functions in `evaluate_model_outputs.py`.

**What it tests:**
- `extract_solution()`: Extract answers from model outputs
  - LaTeX `\boxed{}` format
  - Plain text numbers
  - Edge cases (no answer, multiple numbers, etc.)

- `compute_score()`: Score answers against ground truth
  - Exact matches
  - Normalized comparisons (42 vs 42.0)
  - Number formatting (commas, etc.)

- `estimate_pass_at_k()`: Calculate pass@k metric
  - Various n, c, k combinations
  - Edge cases (all correct, none correct, etc.)
  - Multiple problems with different sample counts

**Run individually:**
```bash
python tests/test_evaluation_functions.py
```

### 3. `run_all_tests.py`
Master test runner that executes all test suites.

**Run all tests:**
```bash
python tests/run_all_tests.py
```

## How to Use

### Quick Start
Run all tests at once:
```bash
cd /path/to/COMP4901B-Homework3
python tests/run_all_tests.py
```

### Debug Individual Functions
If a test fails, run the specific test file to see detailed output:
```bash
# Test only prompt formatting
python tests/test_format_prompts.py

# Test only evaluation functions
python tests/test_evaluation_functions.py
```

## Understanding Test Results

### ✅ PASSED
Your implementation matches the expected output. Good job!

### ❌ FAILED
Your implementation produces different output than expected. The test will show:
- What input was tested
- What output was expected
- What your implementation returned

Review your code and the hint comments in the homework files.

### ⚠️ SKIPPED
The test was skipped (usually because a required dependency is missing, like a tokenizer).
This is okay - the test is optional.

### ⚠️ ERROR
An exception occurred while running your code. Check the traceback for details.

## Test Coverage

These tests check **correctness** but do not guarantee **completeness**:
- ✅ They verify your functions work on common cases
- ✅ They check edge cases
- ❌ They don't test integration (e.g., running full inference)
- ❌ They don't test performance or efficiency

After passing these tests, you should still:
1. Test your code on real data (small subset first)
2. Check that the full pipeline works end-to-end
3. Verify results make sense for the task

## Common Issues

### ImportError: No module named 'vllm'
Some dependencies may not be installed. For the sanity checks, you don't need VLLM installed - we only test the logic functions.

### Tokenizer download fails
If chat template tests fail due to tokenizer download:
- Make sure you have internet connection
- Check your Hugging Face cache directory has space
- The test will skip if tokenizer can't be loaded (marked as ⚠️ SKIPPED)

### Numerical precision errors
For `estimate_pass_at_k()`, we allow small floating-point errors (< 1e-8).
If you're failing with very small differences, check your formula implementation.

## Tips for Passing Tests

1. **Read the docstrings** in the homework files carefully
2. **Follow the hints** provided in the TODO comments
3. **Test incrementally** - implement one function at a time
4. **Print debug info** - use `print()` to see intermediate values
5. **Check the reference** - if stuck, peek at `*_ref.py` files (but try yourself first!)

## Need Help?

If tests are failing and you can't figure out why:
1. Read the detailed error message
2. Check your implementation against the hints
3. Try printing intermediate values to debug
4. Review the reference implementation if available
5. Ask for help in office hours or on the course forum

Good luck! 🚀
