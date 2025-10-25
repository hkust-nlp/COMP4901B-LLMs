#!/usr/bin/env python
"""
Homework helper functions for building loss masks from conversations.

Students implement `conversation_to_features` below, and can run this module
directly to inspect the features produced for the prepared exercise samples.
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


def conversation_to_features(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
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

    # Please implement from here
    # Start with everything masked out; fill tokens as needed per exercise.
    labels = [IGNORE_TOKEN_ID] * len(full_ids)
    attention = [1] * len(full_ids)

    if is_single_turn_conversation(messages):
        # ------------------------------------------------------------------
        # Exercise 1: Single-turn loss mask
        #
        # Goal:
        #   Fill `labels` so that only the assistant response contributes to
        #   the loss for conversations containing exactly one system message
        #   (optional), one user message, and one assistant message.
        #
        # Requirements:
        #   1. Leave system and user tokens masked out (i.e. keep
        #      IGNORE_TOKEN_ID) while copying the assistant tokens into
        #      `labels`.
        #   2. Use `full_ids` (the tokenized whole conversation) and
        #      `prefix_lengths` (cumulative token counts per message) to find
        #      the span that corresponds to the assistant.
        #   3. Assume the input fits within the max length (no truncation yet).
        #
        # Hints:
        #   - `messages` is the normalized chat history in order.
        #   - `prefix_lengths[i]` gives the token count up through
        #     `messages[i]`.
        #
        labels = [IGNORE_TOKEN_ID]*len(full_ids)
        attention = [1]*len(full_ids)

        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                start = prefix_lengths[i-1] if i>0 else 0
                end = prefix_lengths[i]
                labels[start:end] = full_ids[start:end]
                break
            else:
                continue
    else:
        # ------------------------------------------------------------------
        # Exercise 2: Multi-turn loss mask
        #
        # Goal:
        #   Generalize your Exercise 1 logic to support multi-round
        #   conversations. Only assistant utterances should contribute to the
        #   loss; system and user turns should remain masked.
        #
        # Requirements:
        #   1. Support an arbitrary number of (user, assistant) turns that may
        #      be preceded by optional system messages.
        #   2. Handle truncated conversations where `prefix_lengths` may
        #      overshoot `len(full_ids)` (use `min` to stay in bounds).
        #   3. Leave tokens masked (IGNORE_TOKEN_ID) if they should not
        #      contribute to the loss.
        #
        # Hints:
        #   - Reuse or extend the helper logic you wrote for Exercise 1.
        #   - The code that follows assumes `labels` already reflect your
        #     masking decisions.
        #
        labels = [IGNORE_TOKEN_ID]*len(full_ids)
        attention = [1]*len(full_ids)

        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            else:
                start = prefix_lengths[i-1] if i>0 else 0
                end = min(prefix_lengths[i],len(full_ids))
                labels[start:end] = full_ids[start:end]

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
            f"Golden answer file {path} not found. Run generate_golden_answers.py first."
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
        description="Validate conversation_to_features against golden answers."
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
        default=Path("exercise_solutions.json"),
        help="Path to exercise_solutions.json (golden answers).",
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
                features = conversation_to_features(
                    messages,
                    tokenizer,
                    args.model_max_length,
                    args.truncation,
                )
            except NotImplementedError as exc:
                print(
                    f"{exc}. Complete conversation_to_features before running this script."
                )
                return 1

            if features is None:
                raise RuntimeError(
                    f"conversation_to_features returned None for sample id={entry.get('id')}."
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
