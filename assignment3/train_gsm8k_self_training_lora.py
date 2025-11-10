"""Training script for GSM8K self-training.

This script is adapted from train_hw_parallel.py for GSM8K self-training where:
- Data comes from filtered correct examples in JSONL format
- Input field "input" already contains the formatted prompt with chat template
- Output field "model_output" contains the correct solution
- We only train on the output tokens (model_output), not the input prompt

IMPORTANT: Currently only supports single-GPU training. For multi-GPU training,
you need to set CUDA_VISIBLE_DEVICES to a single GPU (e.g., export CUDA_VISIBLE_DEVICES="0")
or modify the script to properly handle distributed training with custom loss function.
"""

from dataclasses import dataclass, field
import gc
import json
import pathlib
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import is_flash_attn_2_available
from tqdm.auto import tqdm

from loss_functions import compute_loss_from_logits
from peft import LoraConfig, TaskType, PeftModel, get_peft_model

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
CACHE_VERSION = 1

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Set True/False to force enabling/disabling flash attention v2. Leave as None for auto detection."
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the GSM8K filtered training data (JSONL)."}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA scaling factor (alpha)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability."})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for LoRA modules."})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of module names to apply LoRA to. "
                    "Supports special keywords 'attn', 'ffn', 'attn_ffn', and 'all'. "
                    "If omitted, a best-effort guess based on the model type is used."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(default=None)


def load_gsm8k_records(data_path: str) -> List[Dict[str, str]]:
    """Load GSM8K filtered data from JSONL.

    Expected format: each line is a JSON object with "input" and "model_output" fields.
    """
    path = pathlib.Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Check required fields
                if "input" not in item or "model_output" not in item:
                    rank0_print(f"Warning: Line {line_num} missing required fields, skipping")
                    continue
                records.append({
                    "input": item["input"],
                    "output": item["model_output"]
                })
            except json.JSONDecodeError as e:
                rank0_print(f"Warning: Line {line_num} failed to parse: {e}")
                continue

    if not records:
        raise ValueError(f"No valid records found in {data_path}")

    rank0_print(f"Loaded {len(records)} examples from {data_path}")
    return records


def _derive_cache_path(data_path: str) -> pathlib.Path:
    path = pathlib.Path(data_path)
    if path.exists():
        return path.with_name(f"{path.stem}_gsm8k_processed.pickle")
    return pathlib.Path.cwd() / "gsm8k_processed.pickle"


def _data_path_signature(data_path: str) -> Dict:
    path = pathlib.Path(data_path)
    if path.exists():
        try:
            stat = path.stat()
            return {
                "data_path": str(path.resolve()),
                "data_mtime": getattr(stat, "st_mtime", None),
                "data_size": getattr(stat, "st_size", None),
            }
        except OSError:
            return {"data_path": str(path)}
    return {"data_path": data_path}


def tokenize_gsm8k_example(
    example: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenize a single GSM8K example.

    The input already contains the formatted prompt with chat template.
    We concatenate input + output and create labels that only train on the output.
    """
    input_text = example["input"]
    output_text = example["output"]

    # Tokenize input and output separately to know where output starts
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)

    # Concatenate
    full_ids = input_ids + output_ids

    # Add EOS token at the end
    if tokenizer.eos_token_id is not None:
        full_ids = full_ids + [tokenizer.eos_token_id]

    # Create labels: mask input, keep output
    labels = [IGNORE_TOKEN_ID] * len(input_ids) + output_ids
    if tokenizer.eos_token_id is not None:
        labels = labels + [tokenizer.eos_token_id]

    # Create attention mask
    attention_mask = [1] * len(full_ids)

    # Truncate if needed (from the left to keep the answer)
    if len(full_ids) > max_length:
        # Truncate from left but try to keep at least some output
        # Calculate how much to truncate
        excess = len(full_ids) - max_length
        # Try to keep full output if possible
        if excess < len(input_ids):
            # Can keep full output, truncate only input
            full_ids = full_ids[excess:]
            labels = labels[excess:]
            attention_mask = attention_mask[excess:]
        else:
            # Must truncate output too (shouldn't happen with correct examples)
            full_ids = full_ids[-max_length:]
            labels = labels[-max_length:]
            attention_mask = attention_mask[-max_length:]

    # Pad if needed
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if len(full_ids) < max_length:
        pad_len = max_length - len(full_ids)
        full_ids = full_ids + [pad_token_id] * pad_len
        labels = labels + [IGNORE_TOKEN_ID] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _serialize_feature(entry: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    return {key: value.cpu().tolist() for key, value in entry.items()}


def _build_dataset_feature(feature: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.long) for key, value in feature.items()}


def _metadata_signature(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    signature = {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "padding_side": getattr(tokenizer, "padding_side", None),
    }
    signature.update(_data_path_signature(data_path))
    return signature


def _try_load_cached_features(
    cache_path: pathlib.Path,
    expected_signature: Dict,
) -> Optional[List[Dict[str, List[int]]]]:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        rank0_print(f"Failed to load cache at {cache_path}: {exc}. Regenerating.")
        return None

    if payload.get("version") != CACHE_VERSION:
        rank0_print(f"Cache version mismatch for {cache_path}. Regenerating.")
        return None

    metadata = payload.get("metadata", {})
    for key, expected_value in expected_signature.items():
        cached_value = metadata.get(key)
        if cached_value != expected_value:
            rank0_print(
                f"Cache metadata mismatch for key '{key}': expected {expected_value}, "
                f"found {cached_value}. Regenerating."
            )
            return None

    features = payload.get("features")
    if not features:
        rank0_print(f"Cache at {cache_path} contained no features. Regenerating.")
        return None
    rank0_print(f"Loaded preprocessed features from {cache_path}.")
    return features


def _preprocess_and_cache_features(
    records: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    cache_path: pathlib.Path,
) -> List[Dict[str, List[int]]]:
    """Preprocess GSM8K examples and cache the results."""
    features = []
    dropped = 0

    progress = tqdm(
        records,
        desc="Tokenizing GSM8K examples",
        disable=local_rank not in (None, 0),
    )

    for example in progress:
        feature = tokenize_gsm8k_example(
            example,
            tokenizer,
            tokenizer.model_max_length,
        )
        if feature is None:
            dropped += 1
            continue
        features.append(_serialize_feature(feature))

    if dropped:
        rank0_print(f"Dropped {dropped} examples during tokenization")

    if not features:
        raise ValueError("No usable examples were found after preprocessing.")

    metadata = _metadata_signature(data_path, tokenizer)
    metadata.update({
        "version": CACHE_VERSION,
        "source_records": len(records),
        "retained": len(features),
        "dropped": dropped,
    })

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"version": CACHE_VERSION, "metadata": metadata, "features": features},
                handle,
            )
        rank0_print(f"Saved preprocessed features to {cache_path}.")
    except Exception as exc:
        rank0_print(f"Failed to save cache at {cache_path}: {exc}.")

    return features


class GSM8KDataset(Dataset):
    """Dataset for GSM8K self-training."""

    def __init__(self, features: List[Dict[str, List[int]]]):
        if not features:
            raise ValueError("No features provided to build the dataset.")
        self._features: List[Dict[str, torch.Tensor]] = [
            _build_dataset_feature(feature) for feature in features
        ]

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        feature = self._features[index]
        return {
            "input_ids": feature["input_ids"].clone(),
            "labels": feature["labels"].clone(),
            "attention_mask": feature["attention_mask"].clone(),
        }


def make_gsm8k_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset for GSM8K self-training."""
    records = load_gsm8k_records(data_args.data_path)
    cache_path = _derive_cache_path(data_args.data_path)
    expected_signature = _metadata_signature(data_args.data_path, tokenizer)

    cached = _try_load_cached_features(cache_path, expected_signature)
    if cached is not None:
        features = cached
    else:
        features = _preprocess_and_cache_features(
            records,
            tokenizer,
            data_args.data_path,
            cache_path,
        )

    # Split into train/eval (98% train, 2% eval)
    np.random.seed(0)
    perm = np.random.permutation(len(features))
    split = max(1, int(len(perm) * 0.98)) if len(perm) > 1 else len(perm)
    train_indices = perm[:split]
    eval_indices = perm[split:] if split < len(perm) else perm[:1]

    train_features = [features[i] for i in train_indices]
    eval_features = [features[i] for i in eval_indices]
    rank0_print(f"#train {len(train_features)}, #eval {len(eval_features)}")

    train_dataset = GSM8KDataset(train_features)
    eval_dataset = GSM8KDataset(eval_features)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class LoRAAdapterManager:
    """
    Manages LoRA adapter application to a model.

    HOMEWORK TASK: Students need to implement the _resolve_lora_target_modules method
    to select appropriate modules for LoRA adaptation based on model architecture.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lora_args: LoraArguments,
        training_args: TrainingArguments,
        local_rank: Optional[int],
    ) -> None:
        self.model = model
        self.lora_args = lora_args
        self.training_args = training_args
        self.local_rank = local_rank

    def apply_adapters(self) -> torch.nn.Module:
        """Apply LoRA adapters to the model."""
        target_modules = self._resolve_lora_target_modules()

        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        # ======================= TODO: Apply LoRA to model =========================
        # Your task: Apply the LoRA configuration to the model
        #
        # The lora_config is already created above with all the parameters.
        # You need to use the appropriate function from the peft library to wrap
        # the model with LoRA adapters.
        #
        # Hint: Look at the imports at the top of this file
        # =======================================================================

        return self.model

    def _resolve_lora_target_modules(self) -> List[str]:
        """
        TODO: HOMEWORK - Implement this method to select target modules for LoRA.

        Background:
        LoRA (Low-Rank Adaptation) works by adding trainable low-rank matrices to
        specific layers in the model. We need to automatically detect which layers
        to adapt based on the model's architecture.

        Your task:
        1. Find all linear layers in the model and collect their names
        2. Detect the model architecture type (e.g., "llama", "opt", "mistral")
        3. Select appropriate modules for LoRA based on the architecture
        4. Validate that the selected modules actually exist in the model
        5. Return a list of module names

        Common layer naming patterns:
        - Attention layers: "q_proj", "k_proj", "v_proj", "o_proj" (or "out_proj")
        - FFN layers: "gate_proj", "up_proj", "down_proj", "fc1", "fc2"
        - Different architectures use different naming conventions

        Hints:
        - self.model.named_modules() gives you all modules with their full names
        - isinstance(module, nn.Linear) identifies linear layers
        - name.split(".")[-1] extracts just the layer name (e.g., "q_proj")
        - self.model.config.model_type tells you the architecture
        - Use rank0_print() to debug and see what's available
        - If user specified lora_target_modules via command line, use those instead

        Returns:
            List of module names to apply LoRA to (e.g., ["q_proj", "k_proj", "v_proj"])
        """
        # ==================== TODO: Implement this method ====================

        # =====================================================================
        valid_targets = []  # Replace with your implementation
        return valid_targets


def _find_lora_adapter_dirs(root: pathlib.Path) -> List[pathlib.Path]:
    """Return directories under `root` that contain saved LoRA adapters."""
    adapter_dirs: List[pathlib.Path] = []
    if root.is_dir() and (root / "adapter_config.json").exists():
        adapter_dirs.append(root)
    checkpoint_dirs = sorted(
        path for path in root.glob("checkpoint-*") if path.is_dir() and (path / "adapter_config.json").exists()
    )
    adapter_dirs.extend(checkpoint_dirs)
    return adapter_dirs


def _merged_dir_for(adapter_dir: pathlib.Path) -> pathlib.Path:
    """Derive a directory path to store the merged full-precision model."""
    return adapter_dir.parent / f"{adapter_dir.name}-merged"


def _merge_adapter_into_base(
    base_model_path: str,
    adapter_dir: pathlib.Path,
    merged_dir: pathlib.Path,
    tokenizer: transformers.PreTrainedTokenizer,
    model_kwargs: Dict,
    save_safetensors: bool,
) -> None:
    """Load base + adapter, merge the weights, and write out a standalone model."""
    if not (adapter_dir / "adapter_config.json").exists():
        rank0_print(f"Skipping merge for {adapter_dir}: adapter_config.json not found.")
        return

    rank0_print(f"Merging LoRA adapter from {adapter_dir} into base model; saving to {merged_dir}")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )
    peft_model = PeftModel.from_pretrained(
        model=base_model,
        model_id=str(adapter_dir),
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    if hasattr(merged_model.config, "use_cache"):
        merged_model.config.use_cache = True
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=save_safetensors,
    )
    tokenizer.save_pretrained(merged_dir)

    # Cleanup
    del merged_model
    del peft_model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank

    flash_available = is_flash_attn_2_available()
    if model_args.flash_attn is True and not flash_available:
        raise RuntimeError(
            "flash_attn was requested but flash_attn_2 is not available in this environment."
        )
    use_flash_attn = flash_available if model_args.flash_attn is None else model_args.flash_attn

    model_kwargs = dict(cache_dir=training_args.cache_dir)
    torch_dtype = None
    if getattr(training_args, "bf16", False):
        torch_dtype = torch.bfloat16
    elif getattr(training_args, "fp16", False):
        torch_dtype = torch.float16
    elif use_flash_attn:
        torch_dtype = torch.bfloat16
        rank0_print("Flash attention requires half precision; defaulting to torch.bfloat16.")
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        rank0_print(f"Loading model in {torch_dtype} precision.")

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        rank0_print("Using flash_attention_2 attention implementation.")

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    except TypeError as type_err:
        if (
            use_flash_attn
            and "unexpected keyword argument 'attn_implementation'" in str(type_err)
        ):
            rank0_print("Falling back to legacy use_flash_attention_2 flag.")
            model_kwargs.pop("attn_implementation", None)
            model_kwargs["use_flash_attention_2"] = True
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs,
            )
        else:
            raise
    model.config.use_cache = False

    lora_manager = LoRAAdapterManager(
        model=model,
        lora_args=lora_args,
        training_args=training_args,
        local_rank=local_rank,
    )
    model = lora_manager.apply_adapters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.model_max_length = training_args.model_max_length

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id for batching.")

    # Note: We don't apply chat template since input already has it
    rank0_print("Using GSM8K self-training mode - input already contains chat template")

    data_module = make_gsm8k_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_loss_func=compute_loss_from_logits,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if local_rank in (None, -1, 0):
        merge_model_kwargs: Dict = {}
        if training_args.cache_dir is not None:
            merge_model_kwargs["cache_dir"] = training_args.cache_dir
        if torch_dtype is not None:
            merge_model_kwargs["torch_dtype"] = torch_dtype
        merge_model_kwargs.setdefault("low_cpu_mem_usage", True)

        adapter_dirs = _find_lora_adapter_dirs(pathlib.Path(training_args.output_dir))
        if adapter_dirs:
            rank0_print("Merging saved LoRA adapters into standalone model weights.")
        for adapter_dir in adapter_dirs:
            merged_dir = _merged_dir_for(adapter_dir)
            _merge_adapter_into_base(
                base_model_path=model_args.model_name_or_path,
                adapter_dir=adapter_dir,
                merged_dir=merged_dir,
                tokenizer=tokenizer,
                model_kwargs=dict(merge_model_kwargs),
                save_safetensors=getattr(training_args, "save_safetensors", True),
            )


if __name__ == "__main__":
    train()
