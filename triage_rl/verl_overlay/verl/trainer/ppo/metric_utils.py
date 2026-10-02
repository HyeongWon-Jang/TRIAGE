# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file has been modified from the original verl source by the authors.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val


def _find_subsequence(seq: list[int], subseq: list[int]) -> int | None:
    n, m = len(seq), len(subseq)
    if m == 0 or n < m:
        return None
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return None


def _safe_get_logprob(logprobs_dict: dict[Any, Any], token_id: int) -> float | None:
    if token_id in logprobs_dict:
        return float(logprobs_dict[token_id])

    token_id_str = str(token_id)
    if token_id_str in logprobs_dict:
        return float(logprobs_dict[token_id_str])

    return None


def _normalize_binary_ground_truth(ground_truth: Any) -> int | None:
    if isinstance(ground_truth, (int, np.integer)):
        if int(ground_truth) in (0, 1):
            return int(ground_truth)
        return None

    if isinstance(ground_truth, str):
        gt = ground_truth.strip()
        if gt in ("0", "1"):
            return int(gt)
        return None

    return None


def compute_binary_positive_probability(
    model_output: str,
    ground_truth: Any = None,
    tokenizer: Any = None,
    top_logprobs: Any = None,
    valid_response_ids: Any = None,
    reward_score: Any = None,
    decision_logprobs: Any = None,
) -> float:
    """Extract P(y=1) for binary classification from generation artifacts.

    Priority:
    1) Decision logprobs from constrained second generation (same as training reward)
    2) Top-logprobs at the token right after "## Final Decision\\n" (or "##Final Decision\\n")
    3) Probabilistic reward recovered from reward_score + ground_truth (only when 0 < score < 1)
    4) Explicit final decision text parsing
    5) Fallback to 0.5
    """
    eps = 1e-12
    markers = ("## Final Decision\n", "##Final Decision\n")

    # Priority 1: Decision logprobs from constrained second generation
    # This matches the probability calculation used during training reward computation.
    if decision_logprobs is not None:
        try:
            lp0 = decision_logprobs.get("logprob_0")
            lp1 = decision_logprobs.get("logprob_1")
            if lp0 is not None and lp1 is not None:
                lp0 = float(lp0)
                lp1 = float(lp1)
                max_lp = max(lp0, lp1)
                p0 = float(np.exp(lp0 - max_lp))
                p1 = float(np.exp(lp1 - max_lp))
                total = p0 + p1 + eps
                if total > 0:
                    p1_norm = np.clip(p1 / total, 0.0, 1.0)
                    return float(p1_norm)
        except (TypeError, ValueError, AttributeError):
            pass

    # Priority 2: Top-logprobs at the decision marker position
    if top_logprobs is not None and valid_response_ids is not None and tokenizer is not None:
        try:
            response_ids = [int(tok) for tok in valid_response_ids]
            if isinstance(top_logprobs, np.ndarray):
                top_logprobs = top_logprobs.tolist()
            token_id_0 = tokenizer.encode("0", add_special_tokens=False)[0]
            token_id_1 = tokenizer.encode("1", add_special_tokens=False)[0]

            for marker in markers:
                marker_ids = tokenizer.encode(marker, add_special_tokens=False)
                pos = _find_subsequence(response_ids, marker_ids)
                if pos is None:
                    continue
                answer_pos = pos + len(marker_ids)
                if answer_pos >= len(top_logprobs):
                    continue
                logprobs_dict = top_logprobs[answer_pos]
                if not isinstance(logprobs_dict, dict):
                    continue

                lp0 = _safe_get_logprob(logprobs_dict, token_id_0)
                lp1 = _safe_get_logprob(logprobs_dict, token_id_1)
                if lp0 is None or lp1 is None:
                    continue

                max_lp = max(lp0, lp1)
                p0 = float(np.exp(lp0 - max_lp))
                p1 = float(np.exp(lp1 - max_lp))
                total = p0 + p1 + eps
                if total > 0:
                    # keep strict numeric bounds
                    p1_norm = np.clip(p1 / total, 0.0, 1.0)
                    return float(p1_norm)
        except Exception:
            pass

    gt = _normalize_binary_ground_truth(ground_truth)
    if reward_score is not None and gt is not None:
        try:
            score = float(reward_score)
            if np.isfinite(score) and 0.0 < score < 1.0:
                return score if gt == 1 else 1.0 - score
        except (TypeError, ValueError):
            pass

    text = model_output if isinstance(model_output, str) else str(model_output)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    for marker in markers:
        if marker in text:
            answer = text.split(marker)[-1].strip()
            if answer and answer[0] in ("0", "1"):
                return float(answer[0] == "1")

    return 0.5


def aggregate_paired_probabilities(
    data_sources: list[Any],
    ground_truths: list[Any],
    positive_probs: list[Any],
    sample_extra_indices: list[int] | None = None,
) -> tuple[list[Any], list[Any], list[float]]:
    """Average probabilities for paired validation samples that share the same patient.

    Groups samples by patient using ``extra_info.index // 2``: each patient has
    two samples with consecutive indices (e.g. 0 & 1, 2 & 3) that differ only
    in reasoning order.  This function averages their predicted probabilities so
    each patient contributes one probability estimate.

    When ``sample_extra_indices`` is not available, falls back to treating every
    sample independently (no aggregation).

    Args:
        data_sources: List of data source identifiers.
        ground_truths: List of ground truth labels.
        positive_probs: List of P(y=1) probabilities.
        sample_extra_indices: List of extra_info.index values for each sample.

    Returns:
        Tuple of (aggregated_data_sources, aggregated_ground_truths, aggregated_probs).
    """
    n = len(positive_probs)

    has_valid_indices = (
        sample_extra_indices is not None
        and len(sample_extra_indices) == n
        and all(idx >= 0 for idx in sample_extra_indices)
    )
    if n == 0 or not has_valid_indices:
        # Cannot group by index; return as-is (no aggregation)
        return data_sources, ground_truths, positive_probs

    # Build groups keyed by patient_id = index // 2
    patient_groups: dict[int, list[int]] = {}
    for i, idx in enumerate(sample_extra_indices):
        patient_id = idx // 2
        patient_groups.setdefault(patient_id, []).append(i)

    agg_ds, agg_gt, agg_prob = [], [], []
    for patient_id, group_indices in patient_groups.items():
        group_probs = []
        for j in group_indices:
            try:
                p = float(positive_probs[j])
            except (TypeError, ValueError):
                continue
            if np.isfinite(p):
                group_probs.append(p)

        if group_probs:
            first = group_indices[0]
            agg_ds.append(data_sources[first])
            agg_gt.append(ground_truths[first])
            agg_prob.append(float(np.mean(group_probs)))

    return agg_ds, agg_gt, agg_prob


def compute_binary_classification_metrics(
    data_sources: list[Any],
    ground_truths: list[Any],
    positive_probs: list[Any],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute per-data-source binary classification metrics from P(y=1)."""
    if not (len(data_sources) == len(ground_truths) == len(positive_probs)):
        raise ValueError(
            "data_sources, ground_truths, and positive_probs must have the same length, got "
            f"{len(data_sources)=}, {len(ground_truths)=}, {len(positive_probs)=}"
        )

    grouped = defaultdict(lambda: {"y_true": [], "y_prob": []})
    for data_source, ground_truth, pos_prob in zip(data_sources, ground_truths, positive_probs, strict=True):
        gt = _normalize_binary_ground_truth(ground_truth)
        if gt is None:
            continue

        try:
            prob = float(pos_prob)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(prob):
            continue

        prob = float(np.clip(prob, 0.0, 1.0))
        source_key = str(data_source)
        grouped[source_key]["y_true"].append(gt)
        grouped[source_key]["y_prob"].append(prob)

    metrics_by_source = {}
    for data_source, values in grouped.items():
        y_true = np.asarray(values["y_true"], dtype=np.int32)
        y_prob = np.asarray(values["y_prob"], dtype=np.float64)
        if y_true.size == 0:
            continue

        y_pred = (y_prob > threshold).astype(np.int32)

        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float("nan")

        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auprc = float("nan")

        metrics_by_source[data_source] = {
            "auroc": auroc,
            "auprc": auprc,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }

    return metrics_by_source
