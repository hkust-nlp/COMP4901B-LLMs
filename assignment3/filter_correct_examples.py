import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Filter correct examples from evaluated results')
    parser.add_argument('--input_jsonl', type=str, required=True,
                        help='Path to input JSONL file with evaluation results (must have score field)')
    parser.add_argument('--output_jsonl', type=str, required=True,
                        help='Path to output JSONL file with only correct examples')
    return parser.parse_args()


def main():
    args = parse_args()

    # Read input JSONL
    print(f"Loading data from {args.input_jsonl}")
    correct_examples = []
    total_count = 0

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            total_count += 1

            # Filter only correct examples (score == 1)
            if item.get('score') == 1:
                correct_examples.append(item)

    # Write correct examples to output JSONL
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for example in correct_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Filtered {len(correct_examples)} correct examples out of {total_count} total examples")
    print(f"Accuracy: {len(correct_examples)/total_count:.2%}")
    print(f"Saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
