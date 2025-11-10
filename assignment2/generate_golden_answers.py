#!/usr/bin/env python
"""
Generate golden loss-mask features for the homework exercises.

This script expects `exercise_samples.json` (produced by
prepare_exercise_samples.py) and emits `exercise_solutions.json` containing the
reference `input_ids`, `labels`, and `attention_mask` tensors for both
single-turn and multi-turn conversations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from train_hw_parallel import IGNORE_TOKEN_ID
from utils import is_single_turn_conversation, QWEN3CHATTEMPLATE


DEFAULT_SAMPLES = Path("exercise_samples.json")
DEFAULT_OUTPUT = Path("exercise_solutions.json")
DEFAULT_MODEL = "SmolLM2-135M"
MODEL_MAX_LENGTH = 2048
TRUNCATION = "left"

def load_tokenizer(model_name: str, model_max_length: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="right",
        model_max_length=model_max_length,
    )

    if "smollm" in model_name.lower():
        tokenizer.chat_template = QWEN3CHATTEMPLATE
        tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.model_max_length = max(tokenizer.model_max_length, model_max_length)
    return tokenizer


def conversation_to_features_reference(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    truncation: str,
) -> Optional[Dict[str, torch.Tensor]]:
    if not messages:
        return None

    try:
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to apply chat template: {exc}") from exc

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

    labels = [IGNORE_TOKEN_ID] * len(full_ids)
    attention = [1] * len(full_ids)
    prev = 0
    for msg, current_len in zip(messages, prefix_lengths):
        current_len = min(current_len, len(full_ids))
        if msg["role"] == "assistant":
            for pos in range(prev, current_len):
                if pos < len(full_ids):
                    labels[pos] = full_ids[pos]
        prev = current_len

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


def _serialize_features(entry: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {key: value.tolist() for key, value in entry.items()}


def _load_samples(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Sample file {path} not found. Run prepare_exercise_samples.py first."
        )
    data = json.loads(path.read_text())
    if "single_turn" not in data or "multi_turn" not in data:
        raise ValueError(f"Sample file {path} is missing required keys.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden features for the exercises.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
        help="Path to exercise_samples.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON file for the golden features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Tokenizer/model name used for tokenization.",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=MODEL_MAX_LENGTH,
        help="Maximum sequence length (should match the homework config).",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        default=TRUNCATION,
        choices=("left", "right"),
        help="Truncation direction for over-length conversations.",
    )
    args = parser.parse_args()

    sample_data = _load_samples(args.samples)
    tokenizer = load_tokenizer(args.model_name, args.model_max_length)

    golden = {
        "meta": {
            "samples_path": str(args.samples),
            "model_name": args.model_name,
            "model_max_length": args.model_max_length,
            "truncation": args.truncation,
        },
        "single_turn": [],
        "multi_turn": [],
    }

    for entry in sample_data["single_turn"]:
        messages = entry["messages"]
        if not is_single_turn_conversation(messages):
            raise ValueError("Single-turn sample does not meet the expected format.")
        features = conversation_to_features_reference(
            messages,
            tokenizer,
            args.model_max_length,
            args.truncation,
        )
        if features is None:
            raise RuntimeError("Golden features returned None for a single-turn sample.")
        golden["single_turn"].append(
            {"id": entry.get("id"), "features": _serialize_features(features)}
        )

    for entry in sample_data["multi_turn"]:
        messages = entry["messages"]
        features = conversation_to_features_reference(
            messages,
            tokenizer,
            args.model_max_length,
            args.truncation,
        )
        if features is None:
            raise RuntimeError("Golden features returned None for a multi-turn sample.")
        golden["multi_turn"].append(
            {"id": entry.get("id"), "features": _serialize_features(features)}
        )

    args.output.write_text(json.dumps(golden, ensure_ascii=False, indent=2))
    print(
        f"Wrote golden features for {len(golden['single_turn'])} single-turn and "
        f"{len(golden['multi_turn'])} multi-turn samples to {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
