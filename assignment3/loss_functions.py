"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.Tensor],
    num_items_in_batch: int,
) -> torch.Tensor:
    """Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        ignore_index: Label id that should be ignored when computing the loss. The
            trainer passes HuggingFace's default ignore index (-100).

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.

    Students should implement this function by computing the cross-entropy loss
    from the raw logits. You may not call `torch.nn.CrossEntropyLoss`; instead,
    derive the loss explicitly using a log-softmax over the vocabulary dimension.
    """

    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    logits = outputs.logits
    return reference_cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def reference_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """Reference implementation of the token-level cross-entropy loss.

    This helper shows one way to compute the language modeling loss manually.
    It mirrors what students are expected to implement in `compute_loss_from_logits`.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {list(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, seq], got {list(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits and labels must align on batch/seq dimensions, got {list(logits.shape)} vs {list(labels.shape)}"
        )

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # Move to log-probabilities in a numerically stable way.
    log_probs = torch.log_softmax(logits, dim=-1)

    flat_log_probs = log_probs.reshape(-1, log_probs.size(-1))
    flat_labels = labels.reshape(-1)

    valid_mask = flat_labels != IGNORE_TOKEN_ID
    if not torch.any(valid_mask):
        # Return a zero loss that maintains gradient connection to logits
        return (flat_log_probs * 0.0).sum()

    valid_log_probs = flat_log_probs[valid_mask]
    valid_labels = flat_labels[valid_mask].long().unsqueeze(1)

    # Gather the log-probability assigned to the ground-truth token at each position.
    token_log_probs = valid_log_probs.gather(dim=1, index=valid_labels).squeeze(1)
    loss = -token_log_probs.sum() / num_items_in_batch
    return loss
