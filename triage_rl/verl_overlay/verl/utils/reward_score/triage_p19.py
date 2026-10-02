# Added to the verl source tree by the authors; not part of upstream verl.
"""Reward function for the TRIAGE binary classification recipe.

Returned dict per sample:

    score: float
        Same as ``sample_reward``. Required by the GDPO reward manager
        contract; not consumed by the GDPO advantage estimator once
        ``algorithm.gdpo_reward_keys`` is set.
    sample_reward: float
        clip( log_likelihood_of_gt_class, sample_reward_clip_low, 0 ) where
        ``log_likelihood = (1 - eps/2) * log p_gt + (eps/2) * log p_other`` and
        ``p_gt`` / ``p_other`` are 2-class-renormalised probabilities derived
        from the constrained-generation logprobs of "0" and "1".
        Format-fail short-circuits to ``sample_reward_clip_low``.
    s_score: float | NaN
        ``logprob_1 - logprob_0`` from the constrained second-stage rollout.
        ``NaN`` when no logprobs are available (format-fail or fallback path).
    prob_1: float
        Renormalised probability of class 1, or 0.5 when no logprobs.
    prob_0: float
        Renormalised probability of class 0, or 0.5 when no logprobs.
    ground_truth_int: int
        ``int(ground_truth)`` ∈ {0, 1}.
    has_logprobs: bool
        True when ``s_score`` is finite (logprobs were available).
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def _find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return None


def _renormalise_two_class(logprob_0, logprob_1):
    """Numerically-stable softmax over the two-class logprob pair."""
    eps = 1e-12
    max_lp = max(logprob_0, logprob_1)
    p0 = float(np.exp(logprob_0 - max_lp))
    p1 = float(np.exp(logprob_1 - max_lp))
    total = p0 + p1 + eps
    p0_norm = float(np.clip(p0 / total, 0.0, 1.0))
    p1_norm = float(np.clip(p1 / total, 0.0, 1.0))
    return p0_norm, p1_norm


def _compute_sample_reward(
    p0_norm: float,
    p1_norm: float,
    ground_truth_int: int,
    sample_reward_clip_low: float,
    label_smoothing_epsilon: float,
) -> float:
    """Soft-targeted log-likelihood reward, clipped to ``[clip_low, 0]``."""
    eps = 1e-12
    if label_smoothing_epsilon > 0.0:
        half_eps = label_smoothing_epsilon / 2.0
        if ground_truth_int == 1:
            p_gt, p_other = p1_norm, p0_norm
        else:
            p_gt, p_other = p0_norm, p1_norm
        log_lik = (1.0 - half_eps) * np.log(p_gt + eps) + half_eps * np.log(p_other + eps)
    else:
        p_gt = p1_norm if ground_truth_int == 1 else p0_norm
        log_lik = np.log(p_gt + eps)
    return float(np.clip(log_lik, sample_reward_clip_low, 0.0))


def check_format(model_output: str) -> bool:
    """Validate the response has exactly the three required headers in one of two orders."""
    if model_output.count("## Rationale for sepsis\n") != 1:
        return False
    if model_output.count("## Rationale for no sepsis") != 1:
        return False
    if model_output.count("## Final Decision\n") != 1:
        return False

    pattern1 = re.compile(
        r"^## Rationale for sepsis\n.*?"
        r"## Rationale for no sepsis\n.*?"
        r"## Final Decision\n.*$",
        re.DOTALL,
    )
    pattern2 = re.compile(
        r"^## Rationale for no sepsis\n.*?"
        r"## Rationale for sepsis\n.*?"
        r"## Final Decision\n.*$",
        re.DOTALL,
    )
    return bool(pattern1.match(model_output) or pattern2.match(model_output))


def _format_fail_result(ground_truth_int: int, sample_reward_clip_low: float) -> dict:
    """Reward emission for a malformed response: clip-low sample reward, no probabilistic info."""
    return {
        "score": sample_reward_clip_low,
        "sample_reward": sample_reward_clip_low,
        "s_score": float("nan"),
        "prob_0": 0.5,
        "prob_1": 0.5,
        "ground_truth_int": ground_truth_int,
        "has_logprobs": False,
    }


def _no_info_result(ground_truth_int: int, sample_reward_clip_low: float) -> dict:
    """Reward emission for the exact-match-failure path: clipped reward, uniform probs."""
    return {
        "score": sample_reward_clip_low,
        "sample_reward": sample_reward_clip_low,
        "s_score": float("nan"),
        "prob_0": 0.5,
        "prob_1": 0.5,
        "ground_truth_int": ground_truth_int,
        "has_logprobs": False,
    }


def _build_result(
    p0_norm: float,
    p1_norm: float,
    s_score: float,
    ground_truth_int: int,
    sample_reward_clip_low: float,
    label_smoothing_epsilon: float,
) -> dict:
    sample_reward = _compute_sample_reward(
        p0_norm,
        p1_norm,
        ground_truth_int,
        sample_reward_clip_low,
        label_smoothing_epsilon,
    )
    return {
        "score": sample_reward,
        "sample_reward": sample_reward,
        "s_score": float(s_score),
        "prob_0": p0_norm,
        "prob_1": p1_norm,
        "ground_truth_int": ground_truth_int,
        "has_logprobs": True,
    }


def compute_score(
    model_output: str,
    ground_truth: str,
    extra_info=None,
    tokenizer=None,
    label_smoothing_epsilon: float = 0.0,
    sample_reward_clip_low: float = -5.0,
    **kwargs,
) -> dict:
    """Compute the per-sample reward dict.

    ``sample_reward`` is the only signal consumed by GRPO/GDPO directly; the
    other keys are plumbing so the trainer can build the sliding-batch separation
    reward without re-decoding the rollout.
    """
    assert ground_truth in ["0", "1"], f"ground_truth must be '0' or '1', got {ground_truth!r}"
    gt_int = int(ground_truth)

    if not check_format(model_output):
        return _format_fail_result(gt_int, sample_reward_clip_low)

    # Path 1: decision logprobs from the constrained second-stage rollout.
    if extra_info is not None:
        decision_logprobs = extra_info.get("decision_logprobs")
        if decision_logprobs is not None:
            lp0 = decision_logprobs.get("logprob_0")
            lp1 = decision_logprobs.get("logprob_1")
            if lp0 is not None and lp1 is not None:
                p0_norm, p1_norm = _renormalise_two_class(lp0, lp1)
                return _build_result(
                    p0_norm,
                    p1_norm,
                    s_score=float(lp1) - float(lp0),
                    ground_truth_int=gt_int,
                    sample_reward_clip_low=sample_reward_clip_low,
                    label_smoothing_epsilon=label_smoothing_epsilon,
                )
            logger.warning(
                "decision_logprobs present but incomplete: logprob_0=%s, logprob_1=%s. "
                "Falling back to top_logprobs path.",
                lp0,
                lp1,
            )

    # Path 2: top logprobs at the decision marker (legacy fallback).
    top_logprobs = None
    response_ids = None
    if extra_info is not None:
        top_logprobs = extra_info.get("top_logprobs")
        response_ids = extra_info.get("valid_response_ids")
    if top_logprobs is not None and response_ids is not None and tokenizer is not None:
        marker_ids = tokenizer.encode("## Final Decision\n", add_special_tokens=False)
        token_id_0 = tokenizer.encode("0", add_special_tokens=False)[0]
        token_id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
        pos = _find_subsequence(response_ids, marker_ids)
        if pos is not None:
            answer_pos = pos + len(marker_ids)
            if answer_pos < len(top_logprobs):
                logprobs_dict = top_logprobs[answer_pos]
                lp0 = logprobs_dict.get(token_id_0)
                lp1 = logprobs_dict.get(token_id_1)
                if lp0 is not None and lp1 is not None:
                    p0_norm, p1_norm = _renormalise_two_class(lp0, lp1)
                    return _build_result(
                        p0_norm,
                        p1_norm,
                        s_score=float(lp1) - float(lp0),
                        ground_truth_int=gt_int,
                        sample_reward_clip_low=sample_reward_clip_low,
                        label_smoothing_epsilon=label_smoothing_epsilon,
                    )
                logger.warning(
                    "Token '0' (id=%d) or '1' (id=%d) not found in top logprobs at decision position. "
                    "Available tokens: %s. Falling back to exact-match reward.",
                    token_id_0,
                    token_id_1,
                    list(logprobs_dict.keys()),
                )

    # Path 3: exact-match fallback. No logprob signal -> the batch-level rewards cannot be computed
    # downstream, so emit s_score=NaN and the trainer skips this sample in pool ops.
    text = model_output
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "## Final Decision\n" in text:
        answer = text.split("## Final Decision\n")[-1].strip()
        if answer and answer[0] in ("0", "1"):
            predicted = int(answer[0])
            sample_reward = 0.0 if predicted == gt_int else sample_reward_clip_low
            return {
                "score": sample_reward,
                "sample_reward": sample_reward,
                "s_score": float("nan"),
                "prob_0": 1.0 if predicted == 0 else 0.0,
                "prob_1": 1.0 if predicted == 1 else 0.0,
                "ground_truth_int": gt_int,
                "has_logprobs": False,
            }

    return _no_info_result(gt_int, sample_reward_clip_low)
