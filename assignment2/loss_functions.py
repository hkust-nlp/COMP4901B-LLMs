"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
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
    return cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """
    Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.
    """
    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    # Calculate vocab probability by log_softmax
    logits_prob = F.log_softmax(logits, dim=-1)

    # Correctly shift to align labels and logits
    sliced_logits_prob = logits_prob[:,:-1,:]
    sliced_labels = labels[:, 1:]

    # Resize tensor to meet the requirement of F.nll_loss()
    bs, seqlen, vocab = logits.shape
    flatten_logits_prob = sliced_logits_prob.contiguous().view(bs*(seqlen-1), vocab)
    flatten_labels = sliced_labels.contiguous().view(bs*(seqlen-1))

    # Calculate thenegative log likelihood loss
    per_token_loss = F.nll_loss(
        flatten_logits_prob,
        flatten_labels,
        ignore_index = IGNORE_TOKEN_ID,
        reduction = "none"
    )

    total_token_loss = per_token_loss.sum()

    # Calculate adverage loss, handling zero divsion case
    if num_items_in_batch > 0:
        adverage_loss = total_token_loss/num_items_in_batch
    else:
        adverage_loss = torch.tensor(0.0, device = logits.device)

    return adverage_loss