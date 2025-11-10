#!/usr/bin/env python3
"""Sanity check for format_prompts implementation.

This script tests whether the student's implementation of format_prompts
matches the expected reference implementation outputs.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data
TEST_QUESTIONS = [
    "If John has 5 apples and gives 2 to Mary, how many apples does he have left?",
    "A train travels 60 miles in 2 hours. What is its average speed?",
    "Sarah has $20 and buys 3 books for $4 each. How much money does she have left?"
]

# Expected outputs for different configurations
# These are pre-computed from the reference implementation

EXPECTED_ZERO_SHOT_NO_CHAT = [
    "If John has 5 apples and gives 2 to Mary, how many apples does he have left?\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "A train travels 60 miles in 2 hours. What is its average speed?\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "Sarah has $20 and buys 3 books for $4 each. How much money does she have left?\nPlease reason step by step, and put your final answer within \\boxed{}."
]

EXPECTED_FEW_SHOT_NO_CHAT_START = "Q: There are 15 trees in the grove."

def test_few_shot_no_chat_template():
    """Test few-shot mode without chat template."""
    print("\n" + "="*70)
    print("TEST 2: Few-shot without chat template")
    print("="*70)

    try:
        from inference_vllm import format_prompts

        result = format_prompts(
            questions=TEST_QUESTIONS[:1],  # Just test one
            use_few_shot=True,
            use_chat_template=False,
            tokenizer=None,
            system_message=None,
            enable_thinking=False
        )

        # Check if few-shot prompt is included
        if len(result) > 0 and EXPECTED_FEW_SHOT_NO_CHAT_START in result[0]:
            print("✅ PASSED: Few-shot prompt contains example problems")
            return True
        else:
            print("❌ FAILED: Few-shot prompt does not contain expected examples")
            print("\nExpected prompt to start with:")
            print(EXPECTED_FEW_SHOT_NO_CHAT_START)
            print("\nGot:")
            print(result[0][:100] if result else "Empty list")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_chat_template():
    """Test with a real tokenizer's chat template."""
    print("\n" + "="*70)
    print("TEST 3: With chat template (using Qwen tokenizer)")
    print("="*70)

    try:
        from inference_vllm import format_prompts
        from transformers import AutoTokenizer

        # Try to load a common model tokenizer
        # We use Qwen2.5 as it's commonly available
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        print(f"Loading tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️  WARNING: Could not load tokenizer ({e})")
            print("Skipping chat template test")
            return None  # Skip test

        result = format_prompts(
            questions=TEST_QUESTIONS[:1],
            use_few_shot=False,
            use_chat_template=True,
            tokenizer=tokenizer,
            system_message="You are a helpful math tutor.",
            enable_thinking=False
        )

        # Check if chat markers are present (varies by model)
        # Common markers: <|im_start|>, <|im_end|>, [INST], </s>, etc.
        prompt = result[0] if result else ""

        has_chat_markers = any(marker in prompt for marker in [
            "<|im_start|>", "<|im_end|>",  # Qwen
            "[INST]", "[/INST]",  # Llama
            "<s>", "</s>",  # Common
            "user", "assistant",  # Generic
        ])

        if has_chat_markers:
            print("✅ PASSED: Chat template markers found in prompt")
            print("\nFormatted prompt preview:")
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            return True
        else:
            print("❌ FAILED: No chat template markers found")
            print("\nGot prompt:")
            print(prompt[:300])
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_message():
    """Test that system message is included when provided."""
    print("\n" + "="*70)
    print("TEST 4: System message inclusion")
    print("="*70)

    try:
        from inference_vllm import format_prompts
        from transformers import AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️  WARNING: Could not load tokenizer ({e})")
            print("Skipping system message test")
            return None

        system_msg = "You are a helpful math tutor."

        result = format_prompts(
            questions=TEST_QUESTIONS[:1],
            use_few_shot=False,
            use_chat_template=True,
            tokenizer=tokenizer,
            system_message=system_msg,
            enable_thinking=False
        )

        prompt = result[0] if result else ""

        # Check if system message content appears in the prompt
        if "math tutor" in prompt or "helpful" in prompt:
            print("✅ PASSED: System message appears to be included")
            return True
        else:
            print("❌ FAILED: System message not found in prompt")
            print("\nExpected to find: 'math tutor' or 'helpful'")
            print("\nGot prompt:")
            print(prompt[:300])
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SANITY CHECK: format_prompts()")
    print("="*70)
    print("\nThis script tests your implementation of the format_prompts function.")
    print("It compares your outputs with expected reference outputs.\n")

    results = []

    # Run tests
    results.append(("Few-shot without chat", test_few_shot_no_chat_template()))
    results.append(("With chat template", test_with_chat_template()))
    results.append(("System message", test_system_message()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)

    for name, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total - skipped} passed")

    if failed == 0:
        print("\n🎉 All tests passed! Your format_prompts implementation looks correct.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review your implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
