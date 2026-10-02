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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _find_subsequence(seq, subseq):
    """Find subseq index with trainer-aligned marker selection.

    Returns the first match when there are 1-2 matches, and the last match
    when there are more than two matches. Returns None if not found.
    """
    n, m = len(seq), len(subseq)
    positions = []
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            positions.append(i)
    if not positions:
        return None
    return positions[-1] if len(positions) > 2 else positions[0]


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_mask = [1] * len(token_output.token_ids)

        raw_top_logprobs = token_output.extra_info.get("top_logprobs")

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        extra = {"turn_scores": [], "tool_rewards": []}
        if raw_top_logprobs is not None:
            extra["top_logprobs"] = raw_top_logprobs[: self.response_length]

        # Store info needed for the deferred decision logprobs phase.
        # The actual second generation is done AFTER all primary generations complete
        # (see AgentLoopWorker.generate_sequences) to avoid vLLM continuous-batching
        # interference between allowed_token_ids and concurrent primary generations.
        decision_marker = self.rollout_config.get("decision_marker", None)
        if decision_marker and token_output.token_ids:
            marker_text = decision_marker if decision_marker.endswith("\n") else decision_marker + "\n"
            marker_ids = self.tokenizer.encode(marker_text, add_special_tokens=False)
            response_ids_list = list(token_output.token_ids)

            pos = _find_subsequence(response_ids_list, marker_ids)
            if pos is not None:
                # Save what the second phase needs: prompt + response up to marker
                extra["_decision_prompt_ids"] = list(prompt_ids) + response_ids_list[: pos + len(marker_ids)]

        output.extra_fields.update(extra)

        return output
