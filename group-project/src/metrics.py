"""
Evaluation metrics for QA tasks.
"""

import re
import string
from typing import List, Union


def normalize_answer(s: str) -> str:
    """Normalize answer string for exact match comparison.

    Lowercase, remove punctuation, articles, and extra whitespace.
    Based on the standard NQ/SQuAD evaluation script.

    Args:
        s: Answer string

    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate Exact Match (EM) score.

    Returns 1.0 if the prediction matches any ground truth answer after
    normalization, otherwise 0.0.

    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        if normalized_prediction == normalize_answer(ground_truth):
            return 1.0

    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate token-level F1 score.

    Computes F1 score between prediction and ground truths at token level.
    Returns the maximum F1 score across all ground truths.

    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers

    Returns:
        Maximum F1 score (0.0 to 1.0)
    """
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()

    if not prediction_tokens:
        return 0.0

    max_f1 = 0.0

    for ground_truth in ground_truths:
        normalized_gt = normalize_answer(ground_truth)
        gt_tokens = normalized_gt.split()

        if not gt_tokens:
            continue

        common_tokens = set(prediction_tokens) & set(gt_tokens)

        if not common_tokens:
            continue

        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(gt_tokens)

        f1 = 2 * (precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1


def extract_answer_from_text(text: str) -> str:
    """Extract answer from model output.

    Looks for answer in <answer> tags first, then falls back to
    using the entire text.

    Args:
        text: Model output text

    Returns:
        Extracted answer string
    """
    # Try to find answer in tags: <answer>...</answer>
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no tags found, use the entire text
    # Remove common prefixes
    text = re.sub(r'^(Answer|The answer is|It is):\s*', '', text, flags=re.IGNORECASE)

    return text.strip()


if __name__ == "__main__":
    # Test the metrics
    print("Testing evaluation metrics...\n")

    # Test 1: Exact match
    pred1 = "The capital of France is Paris."
    gt1 = ["Paris", "paris"]
    em1 = exact_match_score(pred1, gt1)
    print(f"Test 1 - Exact Match")
    print(f"Prediction: {pred1}")
    print(f"Ground truths: {gt1}")
    print(f"EM: {em1} (expected: 1.0)\n")

    # Test 2: No match
    pred2 = "London"
    gt2 = ["Paris"]
    em2 = exact_match_score(pred2, gt2)
    print(f"Test 2 - No Match")
    print(f"Prediction: {pred2}")
    print(f"Ground truths: {gt2}")
    print(f"EM: {em2} (expected: 0.0)\n")

    # Test 3: Answer extraction
    text3 = "Let me think... <answer>42</answer>"
    extracted3 = extract_answer_from_text(text3)
    print(f"Test 3 - Answer Extraction")
    print(f"Text: {text3}")
    print(f"Extracted: {extracted3} (expected: '42')\n")

    # Test 4: F1 score
    pred4 = "The Battle of Waterloo"
    gt4 = ["Battle of Waterloo"]
    f1_4 = f1_score(pred4, gt4)
    print(f"Test 4 - F1 Score")
    print(f"Prediction: {pred4}")
    print(f"Ground truths: {gt4}")
    print(f"F1: {f1_4:.2f} (expected: 1.0)\n")

    print("All tests completed!")
