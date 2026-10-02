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
# Added to the verl source tree by the authors; not part of upstream verl.

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("gdpo")
class GDPORewardManager(RewardManagerBase):
    """GDPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # During the agent loop's per-sample reward call, the agent loop packs
        # the rollout output's ``extra_fields`` dict (which carries
        # ``decision_logprobs`` after phase 2) under ``tool_extra_fields``.
        # Mirror the naive manager and merge it into ``extra_info`` so
        # ``compute_score`` can see ``decision_logprobs``. Without this step,
        # ``s_score`` ends up NaN and the per-sample reward falls back to the
        # exact-match path even when constrained logprobs were available.
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        extra_info["experiment_name"] = self.config.trainer.experiment_name

        decision_logprobs = data_item.non_tensor_batch.get("decision_logprobs", None)
        if decision_logprobs is not None:
            extra_info["decision_logprobs"] = decision_logprobs
        extra_info["valid_response_ids"] = valid_response_ids.tolist()

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        extra_reward_kwargs["tokenizer"] = self.tokenizer

        label_smoothing_epsilon = self.config.reward.get("label_smoothing_epsilon", 0.0)
        if label_smoothing_epsilon > 0.0:
            extra_reward_kwargs["label_smoothing_epsilon"] = label_smoothing_epsilon

        sample_reward_clip_low = self.config.reward.get("sample_reward_clip_low", None)
        if sample_reward_clip_low is not None:
            extra_reward_kwargs["sample_reward_clip_low"] = float(sample_reward_clip_low)
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}