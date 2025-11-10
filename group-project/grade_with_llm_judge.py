"""
Grade student submissions using LLM-as-Judge (DeepSeek).

This script evaluates student responses by comparing them against ground truth answers
using an LLM (DeepSeek) as a judge. The input should be a JSONL file where each line
contains the original NQ question data plus an "llm_response" field with the student's answer.

Expected input format (JSONL):
{
  "id": "nq_validation_0",
  "question": "when was the last time anyone was on the moon?",
  "answers": ["14 December 1972 UTC", "December 1972"],
  "llm_response": "December 1972"  # Student's model output
}

Usage:
    python scripts/grade_with_llm_judge.py \
        --input student_responses.jsonl \
        --model deepseek-chat \
        --base_url https://api.deepseek.com/v1 \
        --api_key YOUR_DEEPSEEK_KEY \
        --output grading_results.json
"""

import json
import argparse
import os
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import time


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


def judge_answer(
    client: OpenAI,
    model_name: str,
    question: str,
    student_answer: str,
    ground_truths: List[str]
) -> Dict[str, Any]:
    """Use LLM to judge if student answer matches ground truth.

    Args:
        client: OpenAI client
        model_name: Model name for judging
        question: The question
        student_answer: Student's predicted answer
        ground_truths: List of acceptable ground truth answers

    Returns:
        Dict with 'correct' (bool), 'explanation' (str), and 'raw_response' (str)
    """
    ground_truths_str = " OR ".join(f'"{gt}"' for gt in ground_truths)

    prompt = f"""You are an evaluation judge. Determine if the student's answer is correct given the ground truth answer(s).

Question: {question}

Student Answer: {student_answer}

Ground Truth Answer(s): {ground_truths_str}

The student answer should be considered CORRECT if it:
1. Contains the same factual information as any of the ground truth answers
2. Is semantically equivalent (e.g., "December 1972" matches "14 December 1972 UTC")
3. Minor formatting differences are acceptable (e.g., "James I" vs "James I.")

The student answer should be considered INCORRECT if:
1. It contains wrong factual information
2. It's too vague or ambiguous
3. It contradicts the ground truth

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation.

Format:
CORRECT/INCORRECT: <brief explanation>"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        result_text = response.choices[0].message.content.strip()

        # Parse result
        if result_text.startswith("CORRECT"):
            correct = True
            explanation = result_text.split(":", 1)[1].strip() if ":" in result_text else result_text
        elif result_text.startswith("INCORRECT"):
            correct = False
            explanation = result_text.split(":", 1)[1].strip() if ":" in result_text else result_text
        else:
            # Default to incorrect if format is unexpected
            correct = False
            explanation = f"Unexpected format: {result_text}"

        return {
            "correct": correct,
            "explanation": explanation,
            "raw_response": result_text
        }

    except Exception as e:
        return {
            "correct": False,
            "explanation": f"Error during judgment: {str(e)}",
            "raw_response": ""
        }


def grade_with_llm_judge(
    responses: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    output_file: str = None
) -> Dict[str, Any]:
    """Grade student responses using LLM as judge.

    Args:
        responses: List of student response dicts
        client: OpenAI client
        model_name: Model name for judging
        output_file: Optional path to save detailed results

    Returns:
        Dict with grading metrics and detailed results
    """
    results = []
    correct_count = 0
    total_count = 0

    print(f"\nGrading {len(responses)} student responses using LLM judge...")

    for resp in tqdm(responses, desc="Grading"):
        question = resp['question']
        student_answer = resp.get('llm_response', '')
        ground_truths = resp['answers']

        # Use LLM to judge
        judgment = judge_answer(client, model_name, question, student_answer, ground_truths)

        if judgment['correct']:
            correct_count += 1
        total_count += 1

        results.append({
            'id': resp.get('id', ''),
            'question': question,
            'student_answer': student_answer,
            'ground_truths': ground_truths,
            'correct': judgment['correct'],
            'explanation': judgment['explanation'],
            'raw_judge_response': judgment['raw_response']
        })

        # Rate limiting
        time.sleep(0.1)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    final_results = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'model_used': model_name,
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
        description="Grade student submissions using LLM-as-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/grade_with_llm_judge.py \\
    --input student_responses.jsonl \\
    --model deepseek-chat \\
    --base_url https://api.deepseek.com/v1 \\
    --api_key YOUR_DEEPSEEK_KEY \\
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
        '--model',
        type=str,
        default='deepseek-chat',
        help='Model name for judging (default: deepseek-chat)'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='https://api.deepseek.com/v1',
        help='Base URL for API endpoint (default: https://api.deepseek.com/v1)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key (required for DeepSeek)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file for grading results'
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        parser.error("--api_key is required (or set OPENAI_API_KEY environment variable)")

    # Initialize client
    client = OpenAI(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url
    )

    # Load student responses
    print(f"Loading student responses from {args.input}...")
    responses = load_student_responses(args.input)

    if len(responses) == 0:
        print("Error: No valid responses found in input file!")
        print("Make sure each line has 'question', 'answers', and 'llm_response' fields.")
        return

    print(f"Loaded {len(responses)} valid responses")

    # Grade responses
    results = grade_with_llm_judge(
        responses=responses,
        client=client,
        model_name=args.model,
        output_file=args.output
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GRADING RESULTS (LLM-AS-JUDGE)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total Questions: {results['total_count']}")
    print(f"Correct Answers: {results['correct_count']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
