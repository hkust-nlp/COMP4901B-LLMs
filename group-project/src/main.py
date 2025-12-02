import argparse
import json
from typing import Optional
from tqdm import tqdm
from agent import agent_loop, generate_no_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", choices=["interactive", "no_search", "search"], default="interactive")
    args = parser.parse_args()

    if args.mode == "interactive":
        try:
            question = input("Enter a question: ").strip()
        except EOFError:
            question = "What is the capital of France?"
        if not question:
            question = "What is the capital of France?"
        answer = agent_loop(question)
        print(answer)
        return

    if not args.dataset or not args.output:
        print("Missing --dataset or --output for batch mode")
        return
    
    with open(args.dataset, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Generating"):
            ex = json.loads(line)
            q = ex["question"]
            if args.mode == "no_search":
                resp = generate_no_search(q)
            else:
                resp = agent_loop(q)
            rec = {
                "id": ex["id"],
                "question": ex["question"],
                "answers": ex.get("answers", []),
                "llm_response": resp,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
