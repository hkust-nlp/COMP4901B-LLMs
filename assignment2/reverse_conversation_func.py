#!/usr/bin/env python
"""
Homework helper functions for building REVERSE loss masks from conversations.

In this exercise, students implement the OPPOSITE masking strategy:
- MASK assistant turns (set labels to IGNORE_TOKEN_ID)
- KEEP user turns for loss calculation

This tests understanding of the masking mechanism by requiring students to
reverse the logic from conversation_func.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from utils import is_single_turn_conversation, QWEN3CHATTEMPLATE

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

DEFAULT_MODEL_NAME = "SmolLM2-135M"
DEFAULT_MODEL_MAX_LENGTH = 2048
DEFAULT_TRUNCATION = "right"
DEFAULT_SAMPLES_PATH = Path("exercise_samples.json")


def reverse_conversation_to_features(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    truncation: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert a conversation to training features with REVERSE loss masking.

    This function implements two different reverse masking strategies:

    1. Single-turn (Part 3):
       - SWAP message ORDER - put assistant message first, user message second
       - Original: [user: "What is Python?"] [assistant: "Python is a language"]
       - After reorder: [assistant: "Python is a language"] [user: "What is Python?"]
       - MASK assistant message (now in first position)
       - KEEP user message (now in second position) for loss calculation

    2. Multi-turn (Part 4):
       - KEEP original message order
       - ADD system message if not present: "You are a good state predictor"
       - MASK assistant turns and system message
       - KEEP user turns for loss calculation

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        tokenizer: The tokenizer to use for encoding
        max_length: Maximum sequence length
        truncation: Truncation direction ('left' or 'right')

    Returns:
        Dictionary with 'input_ids', 'labels', and 'attention_mask' tensors,
        or None if the conversation cannot be processed
    """
    if not messages:
        return None

    # Make a copy to avoid modifying the original
    messages = [msg.copy() for msg in messages]

    # Preprocessing based on conversation type
    if is_single_turn_conversation(messages):
        # Part 3: Swap message ORDER - put assistant first, user second
        # Find user and assistant messages and their indices
        user_idx = None
        assistant_idx = None
        system_msg = None

        for idx, msg in enumerate(messages):
            if msg["role"] == "system":
                system_msg = msg
            elif msg["role"] == "user":
                user_idx = idx
            elif msg["role"] == "assistant":
                assistant_idx = idx

        # Reorder: [system (if exists)] [assistant] [user]
        if user_idx is not None and assistant_idx is not None:
            new_messages = []
            if system_msg is not None:
                new_messages.append(system_msg)
            new_messages.append(messages[assistant_idx])  # assistant first
            new_messages.append(messages[user_idx])       # user second
            messages = new_messages
    else:
        # Part 4: Add system message if not present for multi-turn state prediction
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a good state predictor."}] + messages

    try:
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as exc:  # pragma: no cover - debug helper
        warnings.warn(f"Failed to apply chat template: {exc}")
        return None

    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids.tolist()

    prefix_lengths: List[int] = []
    for end in range(1, len(messages) + 1):
        partial = tokenizer.apply_chat_template(
            messages[:end],
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(partial, torch.Tensor):
            partial = partial.tolist()
        prefix_lengths.append(len(partial))

    # Start with everything masked out; fill tokens as needed per exercise.
    labels = [IGNORE_TOKEN_ID] * len(full_ids)
    attention = [1] * len(full_ids)

    if is_single_turn_conversation(messages):
        # ------------------------------------------------------------------
        # Exercise 3: Single-turn Reverse Loss Masking
        #
        # NOTE: At this point, message ORDER has been SWAPPED!
        # Original: [user: "..."] [assistant: "..."]
        #           position 1    position 2
        #
        # Now:      [assistant: "..."] [user: "..."]
        #           position 1          position 2
        #
        # Goal:
        #   Fill `labels` so that only the USER message (now in position 2)
        #   contributes to the loss.
        #
        # Requirements:
        #   1. MASK assistant message tokens (position 1)
        #   2. KEEP user message tokens (position 2) for loss calculation
        #   3. Use `full_ids` and `prefix_lengths` to identify message spans
        #   4. Assume the input fits within the max length (no truncation yet).
        #
        # Hints:
        #   - After reordering, find the "user" role span (now in position 2)
        #   - This is the OPPOSITE of Exercise 1 (unmask user instead of assistant)
        #   - `prefix_lengths[i]` gives the token count up through `messages[i]`
        #
        # TODO: Implement Exercise 3 here
        raise NotImplementedError(
            "Exercise 3: Please implement single-turn reverse loss masking"
        )
    else:
        # ------------------------------------------------------------------
        # Exercise 4: Multi-turn Reverse Loss Masking
        #
        # NOTE: Message order is NOT changed for multi-turn!
        # A system message "You are a good state predictor." has been added.
        # Original: [user: ...] [assistant: ...] [user: ...] [assistant: ...] ...
        # Now: [system: "You are a good state predictor."] [user: ...] [assistant: ...] ...
        #
        # Goal:
        #   Fill `labels` so that only USER turns contribute to the loss.
        #
        # Requirements:
        #   1. MASK system and assistant role tokens
        #   2. KEEP user role tokens for loss calculation
        #   3. Support arbitrary number of (user, assistant) turns
        #   4. Handle truncation: `prefix_lengths` may exceed `len(full_ids)`
        #      (use `min` to stay in bounds)
        #
        # Hints:
        #   - Similar to Exercise 2, but mask assistant instead of user
        #   - Iterate through messages and unmask only "user" roles
        #   - The system message provides context for the first user turn
        #
        # TODO: Implement Exercise 4 here
        raise NotImplementedError(
            "Exercise 4: Please implement multi-turn reverse loss masking"
        )

    if len(full_ids) > max_length:
        if truncation == "left":
            start = len(full_ids) - max_length
            full_ids = full_ids[start:]
            labels = labels[start:]
            attention = attention[start:]
        else:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
            attention = attention[:max_length]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if len(full_ids) < max_length:
        pad_len = max_length - len(full_ids)
        full_ids = full_ids + [pad_token_id] * pad_len
        labels = labels + [IGNORE_TOKEN_ID] * pad_len
        attention = attention + [0] * pad_len

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def _load_tokenizer(model_name: str, model_max_length: int) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="right",
        model_max_length=model_max_length,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.model_max_length = max(tokenizer.model_max_length, model_max_length)
    return tokenizer


def _load_samples(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Sample file {path} not found. Run prepare_exercise_samples.py first."
        )
    return json.loads(path.read_text())


def _serialize_features(entry: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {key: value.cpu().tolist() for key, value in entry.items()}


def _load_golden(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Golden answer file {path} not found. Run generate_reverse_solutions.py first."
        )
    data = json.loads(path.read_text())
    mapping = {}
    for split in ("single_turn", "multi_turn"):
        mapping[split] = {item["id"]: item["features"] for item in data.get(split, [])}
    metadata = data.get("meta", {})
    return {"meta": metadata, "splits": mapping}


def _compare_features(
    split_name: str,
    payload: Dict,
    golden: Dict[str, Dict],
) -> List[str]:
    mismatches: List[str] = []
    golden_split = golden["splits"].get(split_name, {})
    for entry in payload.get(split_name, []):
        sample_id = entry["id"]
        expected = golden_split.get(sample_id)
        if expected is None:
            mismatches.append(f"sample_id={sample_id}: missing from golden answers")
            continue

        student = entry["features"]
        for key in ("input_ids", "labels", "attention_mask"):
            if student[key] != expected[key]:
                mismatches.append(f"sample_id={sample_id}: field '{key}' differs")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate reverse_conversation_to_features against golden answers."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES_PATH,
        help="Path to exercise_samples.json.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("reverse_exercise_solutions.json"),
        help="Path to reverse_exercise_solutions.json (golden answers).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Tokenizer/model name used for tokenization.",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=DEFAULT_MODEL_MAX_LENGTH,
        help="Maximum sequence length (should match the training config).",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        choices=("left", "right"),
        default=DEFAULT_TRUNCATION,
        help="Truncation direction when sequences exceed the maximum length.",
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Validate multi-turn samples instead of single-turn.",
    )

    args = parser.parse_args()

    sample_data = _load_samples(args.samples)
    golden = _load_golden(args.golden)
    tokenizer = _load_tokenizer(args.model_name, args.model_max_length)

    if "smollm" in args.model_name.lower():
        tokenizer.chat_template = QWEN3CHATTEMPLATE

    payload = {
        "meta": {
            "samples_path": str(args.samples),
            "model_name": args.model_name,
            "model_max_length": args.model_max_length,
            "truncation": args.truncation,
        },
        "single_turn": [],
        "multi_turn": [],
    }

    split_names = ("multi_turn",) if args.multi_turn else ("single_turn",)

    for split_name in split_names:
        for entry in sample_data.get(split_name, []):
            messages = entry["messages"]
            try:
                features = reverse_conversation_to_features(
                    messages,
                    tokenizer,
                    args.model_max_length,
                    args.truncation,
                )
            except NotImplementedError as exc:
                print(
                    f"{exc}. Complete reverse_conversation_to_features before running this script."
                )
                return 1

            if features is None:
                raise RuntimeError(
                    f"reverse_conversation_to_features returned None for sample id={entry.get('id')}."
                )

            payload[split_name].append(
                {"id": entry.get("id"), "features": _serialize_features(features)}
            )

    split = "multi_turn" if args.multi_turn else "single_turn"
    mismatches = _compare_features(split, payload, golden)

    if mismatches:
        print(f"Validation failed for {split}:")
        for item in mismatches:
            print(f"  {item}")
        return 1

    print(f"{split} validation passed. Outputs match the golden answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
