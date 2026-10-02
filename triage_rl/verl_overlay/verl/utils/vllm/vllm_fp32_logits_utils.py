# Added to the verl source tree by the authors; not part of upstream verl.
"""Monkey-patch vLLM's LogitsProcessor so the LM-head GEMM runs in float32.

When the model weights and hidden states are in bfloat16, the LM-head
computation ``hidden_states @ lm_head_weight.T`` inherits BF16 precision.
For tokens whose raw logits are in the [32, 64) range, BF16 resolution is
only 0.25, which quantises logit *differences* to multiples of 0.25 and
severely limits the granularity of probability-based rewards.

The patch casts ``hidden_states`` to float32 before the GEMM so that the
resulting logits (and therefore the downstream logprobs) have full float32
precision.  The transformer body still runs in BF16; only the final
projection is promoted, so the throughput impact is negligible.
"""

import logging
from typing import Optional
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)

_patcher = None


def _get_logits_fp32(
    self,
    hidden_states: torch.Tensor,
    lm_head,
    embedding_bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Drop-in replacement for ``LogitsProcessor._get_logits`` that casts
    hidden states to float32 before the LM-head matmul."""
    # Cast hidden states so that F.linear runs in float32
    hidden_states = hidden_states.to(torch.float32)

    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

    # Gather logits for TP (method defined on the same class)
    logits = self._gather_logits(logits)

    # Remove paddings in vocab (if any)
    if logits is not None:
        logits = logits[..., : self.org_vocab_size]
    return logits


def apply_vllm_fp32_logits_patch() -> None:
    """Activate the float32 LM-head patch (idempotent)."""
    global _patcher
    if _patcher is not None:
        return  # already applied

    target = (
        "vllm.model_executor.layers.logits_processor"
        ".LogitsProcessor._get_logits"
    )
    _patcher = patch(target, _get_logits_fp32)
    _patcher.start()
    logger.info("Applied float32 LM-head logits patch to vLLM LogitsProcessor")
