# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    aggregate_paired_probabilities,
    compute_binary_classification_metrics,
    compute_binary_positive_probability,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction.
        # Also forward ``norm_adv_by_std_in_grpo`` explicitly, because the GDPO estimator
        # accepts it as a kwarg and propagates it down to the per-component
        # ``compute_grpo_outcome_advantage`` calls. Without this, the function's
        # default (True) silently overrides any ``algorithm.norm_adv_by_std_in_grpo=False``
        # config, locking GDPO to standard GRPO normalisation regardless of the user's
        # Dr.GRPO preference.
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
            adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Sliding-batch state for batch-level rewards (TRIAGE recipe).
        # Pool is keyed by patient_id (from extra_info.index // 2) so each patient
        # appears at most once. Entries store (s_patient, y_patient, last_seen_step);
        # patients last seen more than ``sliding_pool_size`` steps ago are evicted.
        reward_cfg = self.config.get("reward", {}) if hasattr(self.config, "get") else {}
        self._sliding_pool_size = int(reward_cfg.get("sliding_pool_size", 5)) if reward_cfg else 5
        self._sliding_pool_warmup_steps = (
            int(reward_cfg.get("sliding_pool_warmup_steps", 5)) if reward_cfg else 5
        )
        self._separation_margin = float(reward_cfg.get("separation_margin", 1.0)) if reward_cfg else 1.0
        self._patient_pool: dict[int, tuple[float, int, int]] = {}

        # Refuse silently-broken combinations involving ce_loss_coef > 0.
        # The decision-token CE term needs (a) ``decision_marker`` set so the trainer
        # can find the position, and (b) ``mask_decision_answer_token=True`` so the
        # mask is populated and the response token is replaced with the ground-truth
        # token id. Either condition missing means CE silently evaluates to 0.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        actor_cfg = self.config.actor_rollout_ref.actor
        ce_coef = float(actor_cfg.get("ce_loss_coef", 0.0) or 0.0)
        if ce_coef > 0.0:
            decision_marker_cfg = rollout_cfg.get("decision_marker", None)
            if not decision_marker_cfg:
                raise ValueError(
                    f"Inconsistent config: actor_rollout_ref.actor.ce_loss_coef={ce_coef} > 0 "
                    "requires actor_rollout_ref.rollout.decision_marker to be set so the "
                    "trainer can locate the decision-answer token. Either set decision_marker "
                    "or set ce_loss_coef=0."
                )
            if not rollout_cfg.get("mask_decision_answer_token", True):
                raise ValueError(
                    f"Inconsistent config: actor_rollout_ref.actor.ce_loss_coef={ce_coef} > 0 "
                    "requires actor_rollout_ref.rollout.mask_decision_answer_token=True so the "
                    "decision-token mask is populated and the response token is replaced with "
                    "the ground-truth token id. Either set mask_decision_answer_token=True or "
                    "set ce_loss_coef=0."
                )

        # ClassBalancedMiniBatchSampler relies on the outer dataloader producing
        # rollout batches whose layout is 50:50 per ``ppo_mini_batch_size`` slice.
        # If the actor's inner mini-batch construction shuffles, that invariant
        # is destroyed before the PPO update sees the data. Assert here so the
        # combination fails loudly at startup instead of silently mixing classes.
        if self.config.data.get("class_balance_sampler", False) and bool(actor_cfg.get("shuffle", False)):
            raise ValueError(
                "Inconsistent config: data.class_balance_sampler=True requires "
                "actor_rollout_ref.actor.shuffle=False so the per-mini-batch 1:1 layout produced "
                "by the sampler survives into the actor's PPO mini-batches. "
                "Set actor_rollout_ref.actor.shuffle=False (this is also the default)."
            )

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(
                self.config.data,
                self.train_dataset,
                ppo_mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            )
        self.train_sampler = train_sampler
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def _json_default(o):
            """Coerce numpy scalars/arrays to native Python types so json.dumps doesn't choke."""
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _gradient_step_debug_print(self, batch: DataProto):
        """Debug print: for one prompt group, show each response's last 20 chars, reward, and advantage."""
        uids = batch.non_tensor_batch["uid"]
        first_uid = uids[0]
        group_indices = [i for i, u in enumerate(uids) if u == first_uid]

        rewards = batch.batch["token_level_scores"].sum(-1).cpu()  # (batch_size,)
        advantages = batch.batch["advantages"].sum(-1).cpu()  # (batch_size,)
        response_mask_sum = batch.batch["response_mask"].sum(-1).cpu().clamp(min=1)  # (batch_size,)
        adv_scalars = advantages / response_mask_sum  # per-sample scalar advantage

        response_ids = batch.batch["responses"]  # (batch_size, response_len)

        gt = batch.non_tensor_batch.get("reward_model", {})
        if isinstance(gt, dict):
            gt_val = gt.get("ground_truth", "N/A")
        else:
            # gt is an array of dicts (one per sample)
            gt_val = gt[0].get("ground_truth", "N/A") if len(gt) > 0 else "N/A"

        lines = [f"[GradientStepDebug][step={self.global_steps}] prompt_uid={first_uid}, gt={gt_val}, n_resp={len(group_indices)}"]
        for rank, idx in enumerate(group_indices):
            resp_text = self.tokenizer.decode(response_ids[idx], skip_special_tokens=True)
            last20 = resp_text[-20:] if len(resp_text) >= 20 else resp_text
            r = rewards[idx].item()
            a = adv_scalars[idx].item()
            lines.append(f"  [resp {rank}] ...{last20} | reward={r:.4f} | advantage={a:.4f}")
        print("\n".join(lines), flush=True)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_positive_probs = []
        sample_extra_indices = []
        sample_repeat_indices = []
        n_repeats = self.config.actor_rollout_ref.rollout.val_kwargs.n

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            pre_repeat_size = len(test_batch)
            test_batch = test_batch.repeat(
                repeat_times=n_repeats, interleave=True
            )

            # Track repeat index for each sample (0, 1, ..., n_repeats-1 per original sample)
            # interleave=True means np.repeat, so repeat indices are [0,1,...,K-1, 0,1,...,K-1, ...]
            batch_repeat_indices = list(range(n_repeats)) * pre_repeat_size
            sample_repeat_indices.extend(batch_repeat_indices)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            prompt_len = test_output_gen_batch.batch["prompts"].shape[-1]
            valid_response_lengths = test_output_gen_batch.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
            top_logprobs_batch = test_output_gen_batch.non_tensor_batch.get("top_logprobs", None)
            decision_logprobs_batch = test_output_gen_batch.non_tensor_batch.get("decision_logprobs", None)

            output_texts = []
            valid_response_ids_batch = []
            for i in range(len(output_ids)):
                valid_len = int(valid_response_lengths[i].item())
                valid_response_ids = output_ids[i][:valid_len].tolist()
                valid_response_ids_batch.append(valid_response_ids)
                output_texts.append(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # Collect extra_info indices for index-based pair grouping
            extra_infos = test_batch.non_tensor_batch.get("extra_info", np.array([]))
            for ei in extra_infos:
                sample_extra_indices.append(ei.get("index", -1) if isinstance(ei, dict) else -1)

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            for i, (response_text, ground_truth, reward_score) in enumerate(zip(output_texts, ground_truths, scores, strict=True)):
                top_logprobs = top_logprobs_batch[i] if top_logprobs_batch is not None else None
                decision_logprobs = decision_logprobs_batch[i] if decision_logprobs_batch is not None else None
                positive_prob = compute_binary_positive_probability(
                    model_output=response_text,
                    ground_truth=ground_truth,
                    tokenizer=self.tokenizer,
                    top_logprobs=top_logprobs,
                    valid_response_ids=valid_response_ids_batch[i],
                    reward_score=reward_score,
                    decision_logprobs=decision_logprobs,
                )
                sample_positive_probs.append(positive_prob)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Build dump data: average scores for patient pairs using extra_info.index,
            # and add individual positive probability as a separate column.
            n_samples = len(sample_scores)

            # Group samples by patient using extra_info.index // 2
            # (each patient has two samples with consecutive indices, e.g. 0&1, 2&3)
            patient_groups = defaultdict(list)  # patient_id -> list of sample positions
            has_valid_indices = len(sample_extra_indices) == n_samples and all(
                idx >= 0 for idx in sample_extra_indices
            )
            if has_valid_indices and n_samples > 0:
                for i, idx in enumerate(sample_extra_indices):
                    patient_groups[idx // 2].append(i)

                # Average scores for each patient group
                dump_scores = list(sample_scores)
                for group_indices in patient_groups.values():
                    avg = sum(sample_scores[i] for i in group_indices) / len(group_indices)
                    for i in group_indices:
                        dump_scores[i] = avg

                # Copy extra infos and average "reward" and "acc" per patient group
                dump_extra = {}
                for key, vals in reward_extra_infos_dict.items():
                    if len(vals) == n_samples and key in ("reward", "acc"):
                        averaged_vals = list(vals)
                        for group_indices in patient_groups.values():
                            avg = sum(vals[i] for i in group_indices) / len(group_indices)
                            for i in group_indices:
                                averaged_vals[i] = avg
                        dump_extra[key] = averaged_vals
                    else:
                        dump_extra[key] = vals

                # Add individual P(gt_class) as "probability" column
                # (P(y=1) is used internally for AUROC/AUPRC; the dump shows P(correct class))
                dump_extra["probability"] = [
                    p if str(gt) == "1" else 1.0 - p
                    for p, gt in zip(sample_positive_probs, sample_gts)
                ]
            else:
                dump_scores = sample_scores
                dump_extra = dict(reward_extra_infos_dict)
                dump_extra["probability"] = [
                    p if str(gt) == "1" else 1.0 - p
                    for p, gt in zip(sample_positive_probs, sample_gts)
                ]

            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=dump_scores,
                reward_extra_infos_dict=dump_extra,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "sample_gts": sample_gts,
                "sample_positive_probs": sample_positive_probs,
                "sample_extra_indices": sample_extra_indices,
                "sample_repeat_indices": sample_repeat_indices,
                "n_repeats": n_repeats,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(
            data_sources=data_sources,
            sample_uids=sample_uids,
            reward_extra_infos_dict=reward_extra_infos_dict,
            sample_turns=sample_turns,
            sample_gts=sample_gts,
            sample_positive_probs=sample_positive_probs,
            sample_extra_indices=sample_extra_indices,
            sample_repeat_indices=sample_repeat_indices,
            n_repeats=n_repeats,
        )

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
        sample_gts=None,
        sample_positive_probs=None,
        sample_extra_indices=None,
        sample_repeat_indices=None,
        n_repeats=1,
    ):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if sample_gts is not None and sample_positive_probs is not None:
            if len(data_sources) == len(sample_gts) == len(sample_positive_probs):
                # Compute classification metrics @n for n = 1, 2, 4, ..., K.
                # @n means: for each patient, use the first n repeats per question
                # ordering (2*n probabilities total per patient), then average.
                n_values = []
                n = 1
                while n <= n_repeats:
                    n_values.append(n)
                    n *= 2

                for at_n in n_values:
                    # Filter to samples with repeat_idx < at_n
                    if sample_repeat_indices and len(sample_repeat_indices) == len(sample_positive_probs):
                        mask = [r < at_n for r in sample_repeat_indices]
                    else:
                        # No repeat info available; include all (equivalent to @K)
                        mask = [True] * len(sample_positive_probs)

                    filt_ds = [v for v, m in zip(data_sources, mask) if m]
                    filt_gt = [v for v, m in zip(sample_gts, mask) if m]
                    filt_prob = [v for v, m in zip(sample_positive_probs, mask) if m]
                    filt_idx = [v for v, m in zip(sample_extra_indices, mask) if m] if sample_extra_indices else None

                    agg_ds, agg_gt, agg_prob = aggregate_paired_probabilities(
                        data_sources=list(filt_ds),
                        ground_truths=filt_gt,
                        positive_probs=filt_prob,
                        sample_extra_indices=filt_idx,
                    )
                    cls_metrics_by_source = compute_binary_classification_metrics(
                        data_sources=agg_ds,
                        ground_truths=agg_gt,
                        positive_probs=agg_prob,
                    )
                    suffix = f"@{at_n}" if n_repeats > 1 else ""
                    for data_source, metric2val in cls_metrics_by_source.items():
                        for metric_name, metric_val in metric2val.items():
                            metric_dict[f"val-core/{data_source}/classification/{metric_name}{suffix}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        _empty_result = {
            "data_sources": [],
            "sample_uids": [],
            "sample_turns": [],
            "sample_gts": [],
            "sample_positive_probs": [],
            "sample_extra_indices": [],
            "sample_repeat_indices": [],
            "n_repeats": 1,
            "reward_extra_infos_dict": {},
        }
        if result_a is None:
            result_a = dict(_empty_result)
        if result_b is None:
            result_b = dict(_empty_result)

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]
        sample_gts = result_a.get("sample_gts", []) + result_b.get("sample_gts", [])
        sample_positive_probs = result_a.get("sample_positive_probs", []) + result_b.get("sample_positive_probs", [])
        sample_extra_indices = result_a.get("sample_extra_indices", []) + result_b.get("sample_extra_indices", [])
        sample_repeat_indices = result_a.get("sample_repeat_indices", []) + result_b.get("sample_repeat_indices", [])
        n_repeats = max(result_a.get("n_repeats", 1), result_b.get("n_repeats", 1))

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(
            data_sources=data_sources,
            sample_uids=sample_uids,
            reward_extra_infos_dict=reward_extra_infos_dict,
            sample_turns=sample_turns,
            sample_gts=sample_gts,
            sample_positive_probs=sample_positive_probs,
            sample_extra_indices=sample_extra_indices,
            sample_repeat_indices=sample_repeat_indices,
            n_repeats=n_repeats,
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        worker_master_port_range = OmegaConf.select(self.config.trainer, "worker_master_port_range")
        if worker_master_port_range is not None:
            wg_kwargs["master_port_range"] = OmegaConf.to_container(worker_master_port_range)
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            lightweight_batch = self._select_log_prob_keys(batch)
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(lightweight_batch)

        return ref_log_prob

    @staticmethod
    def _select_log_prob_keys(batch: DataProto) -> DataProto:
        """Select only the keys needed for log_prob computation to reduce Ray serialization overhead."""
        needed_batch_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "prompts" in batch.batch:
            needed_batch_keys.append("prompts")
        if "response_mask" in batch.batch:
            needed_batch_keys.append("response_mask")
        needed_non_tensor_keys = []
        if "multi_modal_inputs" in batch.non_tensor_batch:
            needed_non_tensor_keys.append("multi_modal_inputs")
        if "uid" in batch.non_tensor_batch:
            needed_non_tensor_keys.append("uid")
        return batch.select(
            batch_keys=needed_batch_keys,
            non_tensor_batch_keys=needed_non_tensor_keys,
        )

    def _compute_sliding_batch_rewards(
        self, batch: DataProto, metrics: dict, reward_extra_infos_dict: dict | None = None
    ) -> None:
        """Compute the batch separation reward from the sliding-batch patient-level pool and
        write them (alongside the per-sample reward) into ``batch.non_tensor_batch``
        so the GDPO advantage estimator can pick them up via ``algorithm.gdpo_reward_keys``.

        Patient-level, de-duplicated:
          - Each patient is identified by ``extra_info.index // 2``.
          - Per current batch: collect all valid rollouts, group by patient_id, average
            their ``s_score`` to get ``s_patient``. ``y_patient`` is the patient's label.
          - Pool is a dict keyed by patient_id, storing ``(s_patient, y_patient, last_seen_step)``.
            Patients not seen in the last ``sliding_pool_size`` training steps are evicted.
          - Comparison set ``A_static`` = patient-level pool ∪ patient-level current batch
            (current overrides pool for shared patient_ids).

        Per-rollout batch separation reward:
          - ``y_i = 1``: ``sep_{i,g} = -(1/|N_A|) Σ_{n in N_A} ℓ(s_{i,g}, s_n)``
          - ``y_i = 0``: ``sep_{i,g} = -(1/|P_A|) Σ_{p in P_A} ℓ(s_p, s_{i,g})``
          where ``N_A``, ``P_A`` are negatives/positives in ``A_static``.

        Warmup: for ``global_steps <= sliding_pool_warmup_steps`` every (i,g) gets
        ``separation_reward = 0``; the current batch is still inserted into the pool.

        Samples whose ``s_score`` is NaN (logprobs unavailable) are excluded from pool
        insertion and receive ``separation_reward = 0``.
        """
        non_tensor = batch.non_tensor_batch

        if "sample_reward" not in non_tensor:
            return

        sample_reward = np.asarray(non_tensor["sample_reward"], dtype=np.float32)
        n_total = len(sample_reward)
        s_score = np.asarray(non_tensor.get("s_score", np.full(n_total, np.nan)), dtype=np.float64)
        gt_int = np.asarray(non_tensor.get("ground_truth_int", np.zeros(n_total, dtype=np.int32)), dtype=np.int32)

        # Patient_id per rollout from extra_info.index // 2.
        patient_id_per_rollout = np.full(n_total, -1, dtype=np.int64)
        extra_info_arr = non_tensor.get("extra_info", None)
        if extra_info_arr is not None:
            for r in range(n_total):
                ei = extra_info_arr[r]
                if isinstance(ei, dict) and ei.get("index") is not None:
                    patient_id_per_rollout[r] = int(ei["index"]) // 2

        valid_mask = np.isfinite(s_score) & (patient_id_per_rollout >= 0)
        metrics["sliding_pool/invalid_ratio"] = float((~valid_mask).mean())
        margin = float(self._separation_margin)

        # Group rollouts by patient_id (current batch). All rollouts of all orderings
        # of one patient contribute to its mean s_score.
        current_patient_means: dict[int, tuple[float, int]] = {}
        per_patient_s_scores: dict[int, list[float]] = {}
        for r in range(n_total):
            if not valid_mask[r]:
                continue
            pid = int(patient_id_per_rollout[r])
            per_patient_s_scores.setdefault(pid, []).append(float(s_score[r]))
            current_patient_means.setdefault(pid, (0.0, int(gt_int[r])))
        for pid, s_list in per_patient_s_scores.items():
            current_patient_means[pid] = (float(np.mean(s_list)), current_patient_means[pid][1])

        # Evict pool entries last seen more than sliding_pool_size steps ago.
        # Keep entries with last_seen >= cutoff so that at step T (post-warmup),
        # A_static = pool ∪ current spans T's pool_size+1 most-recent batches:
        # last_seen ∈ {T - sliding_pool_size, ..., T - 1} ∪ {T}.
        cutoff = self.global_steps - self._sliding_pool_size
        self._patient_pool = {pid: v for pid, v in self._patient_pool.items() if v[2] >= cutoff}

        # Build A_static = pool ∪ current_batch (current overrides pool for shared pids).
        A_static: dict[int, tuple[float, int]] = {pid: (s, y) for pid, (s, y, _) in self._patient_pool.items()}
        for pid, (s_p, y_p) in current_patient_means.items():
            A_static[pid] = (s_p, y_p)

        separation_reward = np.zeros(n_total, dtype=np.float32)

        in_warmup = self.global_steps <= self._sliding_pool_warmup_steps

        if (not in_warmup) and len(A_static) > 0:
            A_pids = list(A_static.keys())
            A_s_static = np.asarray([A_static[pid][0] for pid in A_pids], dtype=np.float64)
            A_y_static = np.asarray([A_static[pid][1] for pid in A_pids], dtype=np.int32)

            pos_mask_static = A_y_static == 1
            neg_mask_static = A_y_static == 0
            P_A_s_static = A_s_static[pos_mask_static]
            N_A_s_static = A_s_static[neg_mask_static]

            def hinge_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                """ℓ(a, b) = max(margin - (a - b), 0)². ``a`` and ``b`` broadcast."""
                gap = margin - (a - b)
                return np.where(gap > 0, gap * gap, 0.0)

            # Per-rollout batch separation reward
            if N_A_s_static.size > 0 or P_A_s_static.size > 0:
                for r in range(n_total):
                    if not valid_mask[r]:
                        continue
                    s_rg = float(s_score[r])
                    if int(gt_int[r]) == 1 and N_A_s_static.size > 0:
                        separation_reward[r] = -float(hinge_sq(np.asarray([s_rg]), N_A_s_static).mean())
                    elif int(gt_int[r]) == 0 and P_A_s_static.size > 0:
                        separation_reward[r] = -float(hinge_sq(P_A_s_static, np.asarray([s_rg])).mean())

            if (~valid_mask).any():
                uid_arr = non_tensor.get("uid", None)
                if uid_arr is not None and len(uid_arr) == n_total:
                    uid_arr = np.asarray(uid_arr, dtype=object)
                    for _u in np.unique(uid_arr):
                        _g = uid_arr == _u
                        _gv = _g & valid_mask
                        if not _gv.any():
                            continue
                        _v = separation_reward[_gv]
                        separation_reward[_g & ~valid_mask] = float(_v.min() - _v.std())

        # Persist current batch's patient-level entries into the pool with
        # last_seen=current_step. Existing entries for the same patient_id are
        # overwritten.
        for pid, (s_p, y_p) in current_patient_means.items():
            self._patient_pool[pid] = (s_p, y_p, self.global_steps)

        non_tensor["separation_reward"] = separation_reward
        non_tensor["sample_reward"] = sample_reward

        # Mirror into reward_extra_infos_dict so the rollout JSONL dump
        # (`_log_rollout_data` -> `_dump_generations`) shows these per row.
        if reward_extra_infos_dict is not None:
            reward_extra_infos_dict["separation_reward"] = separation_reward.tolist()
            reward_extra_infos_dict.setdefault("sample_reward", sample_reward.tolist())
            reward_extra_infos_dict["patient_id"] = patient_id_per_rollout.tolist()
            reward_extra_infos_dict["s_score"] = s_score.tolist()

        metrics["sliding_pool/size"] = len(self._patient_pool)
        metrics["sliding_pool/A_size"] = len(A_static) if not in_warmup else 0
        metrics["sliding_pool/warmup_active"] = float(in_warmup)
        metrics["sliding_pool/n_unique_patients_curr"] = len(current_patient_means)

    def _apply_class_balance_advantage_scaling(self, batch: DataProto, metrics: dict) -> None:
        """Scale advantages per sample so the rollout-batch M:1 class imbalance is
        compensated downstream: minority samples get x2.0, majority samples get x(2.0/M).

        No-op unless the active sampler exposes ``M`` and ``minority_label`` (i.e.
        ``RolloutClassRatioSampler``). Multiplies both ``advantages`` and ``returns``
        in-place.
        """
        sampler = getattr(self, "train_sampler", None)
        if sampler is None or not hasattr(sampler, "M") or not hasattr(sampler, "minority_label"):
            return
        M = float(getattr(sampler, "M"))
        minority_label = int(getattr(sampler, "minority_label"))
        minority_weight = 2.0
        majority_weight = 2.0 / M

        reward_model = batch.non_tensor_batch.get("reward_model", None)
        if reward_model is None:
            return

        import torch as _torch

        batch_size = batch.batch["advantages"].shape[0]
        weights = _torch.empty(batch_size, dtype=batch.batch["advantages"].dtype, device=batch.batch["advantages"].device)
        for i in range(batch_size):
            entry = reward_model[i] if i < len(reward_model) else None
            try:
                gt_int = int(str(entry.get("ground_truth")).strip()) if entry is not None else None
            except (AttributeError, ValueError, TypeError):
                gt_int = None
            if gt_int is None:
                weights[i] = 1.0
            elif gt_int == minority_label:
                weights[i] = minority_weight
            else:
                weights[i] = majority_weight

        batch.batch["advantages"] = batch.batch["advantages"] * weights[:, None]
        if "returns" in batch.batch:
            batch.batch["returns"] = batch.batch["returns"] * weights[:, None]

        metrics["batch_balance/per_sample_weight_mean"] = float(weights.mean().item())
        metrics["batch_balance/per_sample_weight_min"] = float(weights.min().item())
        metrics["batch_balance/per_sample_weight_max"] = float(weights.max().item())
        metrics["batch_balance/M"] = M

    def _mask_decision_answer_tokens(self, batch: DataProto):
        """Locate the single answer token that follows ``decision_marker``
        ("## Final Decision\\n") and prepare it for the new loss decomposition:

          - Replace ``responses[i, answer_pos]`` and the corresponding position
            in ``input_ids`` with the ground-truth token id ("0" or "1"). This
            makes ``log_prob[i, answer_pos]`` evaluate to ``log P_actor(gt | prefix)``
            in the actor's forward pass, which is exactly what the new
            ``ce_loss_coef * cross_entropy`` term needs.
          - Build a per-sample ``decision_token_mask`` of shape
            ``(batch_size, response_length)`` with 1 at ``answer_pos`` and 0 elsewhere,
            stored at ``batch.batch["decision_token_mask"]``. The actor will use
            this to (a) exclude the position from the PPO surrogate loss and
            (b) compute the cross-entropy term ``-log_prob`` at that position.

        ``response_mask`` is **NOT** mutated; KL and entropy continue to apply
        to the full response (including the decision token). The PPO surrogate
        is masked at the actor via ``policy_mask = response_mask * (1 - decision_token_mask)``.

        No-op (writes a zero ``decision_token_mask``) when either
        ``actor_rollout_ref.rollout.decision_marker`` is unset or
        ``actor_rollout_ref.rollout.mask_decision_answer_token`` is False.
        """
        responses = batch.batch["responses"]
        # Always populate decision_token_mask so the actor sees a consistent key.
        batch.batch["decision_token_mask"] = torch.zeros_like(responses, dtype=torch.float32)

        rollout_config = self.config.actor_rollout_ref.rollout
        decision_marker = rollout_config.get("decision_marker", None)
        if not decision_marker:
            return
        if not rollout_config.get("mask_decision_answer_token", True):
            return

        marker_text = decision_marker if decision_marker.endswith("\n") else decision_marker + "\n"
        marker_ids = self.tokenizer.encode(marker_text, add_special_tokens=False)
        marker_len = len(marker_ids)
        token_id_0 = self.tokenizer.encode("0", add_special_tokens=False)[0]
        token_id_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]
        valid_answer_token_ids = {token_id_0, token_id_1}
        gt_token_map = {"0": token_id_0, "1": token_id_1}

        input_ids = batch.batch["input_ids"]
        prompt_length = input_ids.shape[1] - responses.shape[1]
        decision_token_mask = batch.batch["decision_token_mask"]
        batch_size = responses.shape[0]

        reward_model_arr = batch.non_tensor_batch.get("reward_model", None)
        n_masked = 0
        n_gt_missing = 0
        for i in range(batch_size):
            resp_ids = responses[i].tolist()

            marker_positions = []
            for j in range(len(resp_ids) - marker_len + 1):
                if resp_ids[j : j + marker_len] == marker_ids:
                    marker_positions.append(j)

            if not marker_positions:
                continue

            # Resolve gt token for this sample.
            entry = reward_model_arr[i] if reward_model_arr is not None and i < len(reward_model_arr) else None
            try:
                gt_str = str(entry.get("ground_truth")).strip() if entry is not None else None
            except (AttributeError, ValueError, TypeError):
                gt_str = None
            if gt_str is None or gt_str not in gt_token_map:
                n_gt_missing += 1
                continue
            gt_token_id = gt_token_map[gt_str]

            for pos in reversed(marker_positions):
                answer_pos = pos + marker_len
                if answer_pos >= responses.shape[1]:
                    continue
                if resp_ids[answer_pos] not in valid_answer_token_ids:
                    continue

                # Replace the sampled "0"/"1" with the ground-truth token id so the
                # actor's log_prob at this position becomes log P(gt | prefix).
                responses[i, answer_pos] = gt_token_id
                input_ids[i, prompt_length + answer_pos] = gt_token_id
                decision_token_mask[i, answer_pos] = 1.0
                n_masked += 1
                break

        if n_masked > 0 or n_gt_missing > 0:
            print(
                "[_mask_decision_answer_tokens] "
                f"Marked decision answer token in {n_masked} / {batch_size} samples"
                + (f" (skipped {n_gt_missing} for missing ground_truth)." if n_gt_missing else ".")
            )
        elif float(self.config.actor_rollout_ref.actor.get("ce_loss_coef", 0.0) or 0.0) > 0.0:
            # Mask is enabled and CE is requested, but no rollout had the decision marker.
            # This silently makes the CE term 0 for the entire batch; warn loudly.
            print(
                "[_mask_decision_answer_tokens] WARNING: ce_loss_coef > 0 but the decision "
                f"marker was not found in any of {batch_size} rollouts. The CE term will "
                "evaluate to 0 for this step. Check that the model is producing the "
                f"`{decision_marker}\\n` format."
            )

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            lightweight_batch = self._select_log_prob_keys(batch)
            old_log_prob = self.actor_rollout_wg.compute_log_prob(lightweight_batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Compute the sliding-batch separation reward from the per-prompt s_i pool
                    # and write the three reward components into batch.non_tensor_batch
                    # so GDPO's compute_advantage picks them up via gdpo_reward_keys.
                    # Also mirror into reward_extra_infos_dict so they appear in
                    # the rollout JSONL dump.
                    self._compute_sliding_batch_rewards(batch, metrics, reward_extra_infos_dict)

                    # Zero response_mask at the single decision token so the sampled
                    # "0"/"1" contributes no gradient. The reward still depends on
                    # the decision via the constrained second-stage logprobs.
                    self._mask_decision_answer_tokens(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Compensate the natural M:1 class imbalance in the rollout batch
                        # by scaling each sample's advantage. Minority samples get x2,
                        # majority samples get x(2/M), so the average per-class gradient
                        # contribution is balanced. Applied AFTER compute_advantage so that
                        # GDPO's terminal masked_whiten does not erase the multiplier.
                        self._apply_class_balance_advantage_scaling(batch, metrics)

                    # Debug print: sample one prompt group and show responses, rewards, advantages
                    if self.config.trainer.get("gradient_step_debug_print", False):
                        self._gradient_step_debug_print(batch)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    rollout_data_freq = self.config.trainer.get("rollout_data_freq", 0)
                    if rollout_data_dir and (rollout_data_freq <= 0 or self.global_steps<=5 or self.global_steps % rollout_data_freq == 0):
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # Per-component reward metrics. Emit the standard mean/std/max/min for
                # every key declared in ``algorithm.gdpo_reward_keys`` (when GDPO is on)
                # PLUS extra slices (zero-std-group ratio, per-class mean/std) for the
                # triage 3-component rewards whenever they are in the batch.
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))

                # Extra per-reward slices: emit for sample_reward / separation_reward
                # regardless of whether they're in gdpo_reward_keys, since they're all
                # tracked by the sliding-batch helper.
                extra_keys = ("sample_reward", "separation_reward")
                uid_arr = batch.non_tensor_batch.get("uid", None)
                gt_arr_raw = batch.non_tensor_batch.get("ground_truth_int", None)
                gt_arr = np.asarray(gt_arr_raw, dtype=np.int32) if gt_arr_raw is not None else None
                for key in extra_keys:
                    if key not in batch.non_tensor_batch:
                        continue
                    vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                    # Always emit the base 4 metrics so they appear consistently even
                    # when the key isn't part of gdpo_reward_keys (i.e. CE-mode training
                    # where sample_reward isn't fed into the advantage estimator).
                    metrics.setdefault(f"gdpo/{key}/mean", float(np.mean(vals)))
                    metrics.setdefault(f"gdpo/{key}/std", float(np.std(vals)))
                    metrics.setdefault(f"gdpo/{key}/max", float(np.max(vals)))
                    metrics.setdefault(f"gdpo/{key}/min", float(np.min(vals)))

                    # zero_std_group_ratio: fraction of prompt groups (keyed by uid)
                    # whose rollout-level reward values are all identical.
                    if uid_arr is not None and len(uid_arr) == len(vals):
                        groups: dict = {}
                        for i, u in enumerate(uid_arr):
                            groups.setdefault(u, []).append(vals[i])
                        if groups:
                            n_zero_std = sum(1 for g in groups.values() if float(np.std(g)) == 0.0)
                            metrics[f"gdpo/{key}/zero_std_group_ratio"] = n_zero_std / len(groups)

                    # Per-class slices.
                    if gt_arr is not None and len(gt_arr) == len(vals):
                        pos_mask = gt_arr == 1
                        neg_mask = gt_arr == 0
                        if pos_mask.any():
                            pos_vals = vals[pos_mask]
                            metrics[f"gdpo/{key}/positive/mean"] = float(np.mean(pos_vals))
                            metrics[f"gdpo/{key}/positive/std"] = float(np.std(pos_vals))
                        if neg_mask.any():
                            neg_vals = vals[neg_mask]
                            metrics[f"gdpo/{key}/negative/mean"] = float(np.mean(neg_vals))
                            metrics[f"gdpo/{key}/negative/std"] = float(np.std(neg_vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
