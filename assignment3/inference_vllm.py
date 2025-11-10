#!/usr/bin/env python3
"""VLLM-based inference script for GSM8K benchmark.

This script loads a model using VLLM, generates responses for GSM8K test set,
and saves the results in jsonl format.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Few-shot prompt with 8 examples
FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""

ZERO_SHOT_PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

def extract_ground_truth(text):
    """Extract the numerical answer from GSM8K answer text.

    Args:
        text: Answer text in format "explanation #### number"

    Returns:
        Extracted numerical answer
    """
    return text.split('####')[-1].strip()


def load_gsm8k_data(split: str = 'test') -> List[Dict]:
    """Load the GSM8K dataset.

    Args:
        split: Dataset split to load ('train' or 'test')

    Returns:
        List of dictionaries with question and answer
    """
    logger.info(f"Loading GSM8K dataset (split: {split})...")
    dataset = load_dataset('gsm8k-local', 'main', split=split, cache_dir=os.path.join(os.path.dirname(__file__), '.hf_cache'))
    logger.info(f"Loaded {len(dataset)} examples from GSM8K {split} set")
    return dataset


def format_prompts(
    questions: List[str],
    use_few_shot: bool = False,
    use_chat_template: bool = True,
    tokenizer: AutoTokenizer = None,
    system_message: str = None,
    enable_thinking: bool = False
) -> List[str]:
    """Format GSM8K questions into prompts.

    Args:
        questions: List of GSM8K questions
        use_few_shot: Whether to use few-shot prompting (8 examples)
        use_chat_template: Whether to apply chat template
        tokenizer: HuggingFace tokenizer for chat template
        system_message: Optional system message for chat template
        enable_thinking: Whether to enable thinking mode for Qwen3 models (default: False)

    Returns:
        List of formatted prompts ready for inference
    """
    formatted_prompts = []

    # ======================= TODO: Implement this method =========================
    # Your task: Format each question into a prompt suitable for the model
    #
    # Steps:
    # 1. For each question:
    #    a. Choose the appropriate prompt template based on use_few_shot flag
    #       - FEW_SHOT_PROMPT: includes 8 example Q&A pairs (defined above)
    #       - ZERO_SHOT_PROMPT: just the question with instruction (defined above)
    #    b. Format the template with the question using .format(question=...)
    #
    # 2. If use_chat_template is True and tokenizer is provided:
    #    - Create a messages list in chat format (list of dicts with "role" and "content")
    #    - If system_message is provided, add {"role": "system", "content": system_message}
    #    - Add the formatted prompt as {"role": "user", "content": formatted_prompt}
    #    - Apply tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #    - Wrap in try-except to handle models that don't support chat templates
    #
    # 3. Otherwise: use the formatted prompt directly (without chat template)
    #
    # Hint: The chat template transforms messages into the model's expected format
    # (e.g., "<|im_start|>user\n{content}<|im_end|>" for Qwen models)
    # =======================================================================
    return formatted_prompts


def run_inference(
    model_path: str,
    output_path: str,
    tensor_parallel_size: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    batch_size: int = 32,
    system_message: str = None,
    gpu_memory_utilization: float = 0.9,
    use_few_shot: bool = False,
    use_chat_template: bool = True,
    enable_thinking: bool = False,
    n_rollouts: int = 1,
    split: str = 'test',
    n_queries: int = -1,
) -> None:
    """Run VLLM inference on GSM8K dataset.

    Args:
        model_path: Path to the trained model
        output_path: Path to save output responses in jsonl format
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (-1 to disable)
        batch_size: Batch size for inference (not used in VLLM)
        system_message: Optional system message for chat template
        gpu_memory_utilization: GPU memory utilization fraction
        use_few_shot: Whether to use few-shot prompting (8 examples)
        use_chat_template: Whether to apply chat template
        enable_thinking: Whether to enable thinking mode for Qwen3 models (default: False)
        n_rollouts: Number of rollouts (generations) per question (default: 1)
        split: Dataset split to use ('train' or 'test', default: 'test')
        n_queries: Number of queries to use from dataset (-1 for all, default: -1)
    """
    logger.info("="*80)
    logger.info("Starting VLLM Inference for GSM8K")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset split: {split}")
    logger.info(f"Mode: {'Few-shot (8 examples)' if use_few_shot else 'Zero-shot'}")
    logger.info(f"Chat template: {'Enabled' if use_chat_template else 'Disabled'}")
    logger.info(f"Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-p: {top_p}")
    logger.info(f"Top-k: {top_k}")
    logger.info(f"Number of rollouts per question: {n_rollouts}")

    # Load GSM8K dataset
    dataset = load_gsm8k_data(split=split)
    total_queries = len(dataset)

    # Limit to n_queries if specified
    if n_queries > 0 and n_queries < total_queries:
        dataset = dataset.select(range(n_queries))
        logger.info(f"Using subset of {n_queries} queries (from {total_queries} total)")
    else:
        logger.info(f"Using full dataset ({total_queries} queries)")

    questions = [item["question"] for item in dataset]
    gold_answers = [item["answer"] for item in dataset]

    # Load tokenizer if using chat template
    tokenizer = None
    if use_chat_template:
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Format prompts
    logger.info("Formatting prompts...")
    formatted_prompts = format_prompts(
        questions,
        use_few_shot=use_few_shot,
        use_chat_template=use_chat_template,
        tokenizer=tokenizer,
        system_message=system_message,
        enable_thinking=enable_thinking
    )

    # Initialize VLLM
    logger.info("Initializing VLLM engine")
    # ======================= TODO: Implement VLLM inference =========================
    # Your task: Use VLLM to generate model outputs for the formatted prompts
    #
    # Steps:
    # 1. Initialize the VLLM LLM engine with appropriate parameters
    #    (model path, tensor parallelism, memory utilization, etc.)
    #
    # 2. Configure sampling parameters:
    #    - Consider the difference between greedy decoding (temperature=0.0) and
    #      sampling-based generation (temperature>0.0)
    #    - When temperature=0.0: deterministic, always picks highest probability token
    #    - When temperature>0.0: stochastic, samples from probability distribution
    #    - For multiple rollouts (n_rollouts>1), you typically want temperature>0.0
    #      to get diverse outputs
    #
    # 3. Generate outputs using the LLM engine with the formatted prompts
    #    Store the results in a variable named "outputs"
    #
    # Hint: Check the VLLM documentation for LLM and SamplingParams classes
    # can refer to https://docs.vllm.ai/en/stable/getting_started/quickstart.html
    # =======================================================================


    # Save results
    logger.info(f"Saving results to {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rollouts = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (prompt, output) in enumerate(zip(formatted_prompts, outputs)):
            # Iterate through all rollouts for this question
            for rollout_idx, completion in enumerate(output.outputs):
                # Extract model output based on mode
                if use_few_shot:
                    model_output = completion.text.strip().split("\n\n")[0].strip()
                else:
                    model_output = completion.text

                result = {
                    "idx": i,  # Original question index
                    "question": questions[i],
                    "input": prompt,
                    "gold_answer": gold_answers[i],
                    "gt": extract_ground_truth(gold_answers[i]),
                    "model_output": model_output
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                total_rollouts += 1

    logger.info("="*80)
    logger.info(f"Inference complete! Generated {total_rollouts} total rollouts from {len(outputs)} questions")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="VLLM inference for GSM8K benchmark"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output responses (jsonl format)"
    )

    # Optional arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (-1 to disable)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (not used in VLLM, for compatibility)"
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        help="Optional system message to prepend to prompts"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (0.0-1.0)"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use few-shot prompting with 8 examples (default: zero-shot)"
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template and use raw prompts"
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models (default: disabled)"
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="Number of rollouts (generations) per question (default: 1)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=-1,
        help="Number of queries to use from dataset (-1 for all, default: -1)"
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        output_path=args.output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        system_message=args.system_message,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_few_shot=args.use_few_shot,
        use_chat_template=not args.no_chat_template,
        enable_thinking=args.enable_thinking,
        n_rollouts=args.n_rollouts,
        split=args.split,
        n_queries=args.n_queries,
    )


if __name__ == "__main__":
    main()
