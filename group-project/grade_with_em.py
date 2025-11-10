"""
Grade student submissions using Exact Match (EM) and F1 Score.

This script evaluates student responses by computing Exact Match and F1 scores
against ground truth answers. The input should be a JSONL file where each line
contains the original NQ question data plus an "llm_response" field with the student's answer.

Expected input format (JSONL):
{
  "id": "nq_validation_0",
  "question": "when was the last time anyone was on the moon?",
  "answers": ["14 December 1972 UTC", "December 1972"],
  "llm_response": "December 1972"  # Student's model output
}

Usage:
    python scripts/grade_with_em.py \
        --input student_responses.jsonl \
        --output grading_results.json
"""

import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from src.metrics import exact_match_score, f1_score, extract_answer_from_text


def load_student_responses(input_file: str) -> List[Dict[str, Any]]:
    """Load student responses from JSONL file.

    Expected format: Each line should have 'question', 'answers', and 'llm_response' fields.
    """
    responses = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Validate required fields
                    if 'question' not in data:
                        print(f"Warning: Line {line_num} missing 'question' field, skipping")
                        continue
                    if 'answers' not in data:
                        print(f"Warning: Line {line_num} missing 'answers' field, skipping")
                        continue
                    if 'llm_response' not in data:
                        print(f"Warning: Line {line_num} missing 'llm_response' field, skipping")
                        continue
                    responses.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON: {e}")
                    continue
    return responses


def grade_with_em_f1(
    responses: List[Dict[str, Any]],
    output_file: str = None
) -> Dict[str, Any]:
    """Grade student responses using Exact Match and F1 score.

    Args:
        responses: List of student response dicts
        output_file: Optional path to save detailed results

    Returns:
        Dict with grading metrics and detailed results
    """
    results = []
    em_scores = []
    f1_scores = []

    print(f"\nGrading {len(responses)} student responses using EM and F1...")

    for resp in tqdm(responses, desc="Grading"):
        question = resp['question']
        student_answer = resp.get('llm_response', '')
        ground_truths = resp['answers']

        # Extract answer (normalize text)
        extracted_answer = extract_answer_from_text(student_answer)

        # Calculate metrics
        em = exact_match_score(extracted_answer, ground_truths)
        f1 = f1_score(extracted_answer, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

        results.append({
            'id': resp.get('id', ''),
            'question': question,
            'student_answer': student_answer,
            'extracted_answer': extracted_answer,
            'ground_truths': ground_truths,
            'exact_match': em,
            'f1_score': f1
        })

    # Calculate average metrics
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    final_results = {
        'exact_match': avg_em,
        'f1_score': avg_f1,
        'total_count': len(responses),
        'detailed_results': results
    }

    # Save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_file}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Grade student submissions using Exact Match and F1 Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/grade_with_em.py \\
    --input student_responses.jsonl \\
    --output grading_results.json
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to student responses JSONL file (must contain "question", "answers", and "llm_response" fields)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file for grading results'
    )

    args = parser.parse_args()

    # Load student responses
    print(f"Loading student responses from {args.input}...")
    responses = load_student_responses(args.input)

    if len(responses) == 0:
        print("Error: No valid responses found in input file!")
        print("Make sure each line has 'question', 'answers', and 'llm_response' fields.")
        return

    print(f"Loaded {len(responses)} valid responses")

    # Grade responses
    results = grade_with_em_f1(
        responses=responses,
        output_file=args.output
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GRADING RESULTS (EXACT MATCH & F1)")
    print("=" * 60)
    print(f"Total Questions: {results['total_count']}")
    print(f"Exact Match (EM): {results['exact_match']:.2%}")
    print(f"F1 Score: {results['f1_score']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
