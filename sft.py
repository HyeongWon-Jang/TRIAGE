# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# This file has been modified from the original version.

import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys

import datasets
import numpy as np
import transformers
import torch
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, TrlParser, ModelConfig
from trl.trainer import utils as trl_utils
from trl.trainer import sft_trainer as trl_sft_trainer

from configs import ScriptArguments, SFTConfig
from utils import get_model, get_tokenizer, init_wandb_training


logger = logging.getLogger(__name__)

from torch.utils.data import Sampler


class BalancedStreamSampler(Sampler):
    def __init__(self, labels, window_size: int, minority_multiplier: int = 3, seed: int = 42):
        self.labels = np.asarray(labels, dtype=int)
        self.window_size = int(window_size)
        self.minority_multiplier = int(minority_multiplier)
        self.seed = int(seed)

        if self.window_size % 2 != 0:
            raise ValueError("window_size must be even for 1:1 windows.")
        self.half = self.window_size // 2

        idx_0 = np.where(self.labels == 0)[0]
        idx_1 = np.where(self.labels == 1)[0]
        if len(idx_0) == 0 or len(idx_1) == 0:
            raise ValueError("Both classes must be present.")

        if len(idx_0) >= len(idx_1):
            self.majority_idx = idx_0
            self.minority_idx = idx_1
        else:
            self.majority_idx = idx_1
            self.minority_idx = idx_0

        self.expanded_minority = np.concatenate([self.minority_idx] * self.minority_multiplier, axis=0)

        # epoch state
        self.epoch = 0
        self._cached_flat = None
        self._n_windows = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._cached_flat = None

    def _build_stream_for_epoch(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        I0 = self.majority_idx.copy()
        I1 = self.expanded_minority.copy()
        rng.shuffle(I0)
        rng.shuffle(I1)

        K0 = len(I0) // self.half
        K1 = len(I1) // self.half
        n_windows = min(K0, K1)
        if n_windows <= 0:
            raise ValueError("Not enough samples to form a single balanced window.")
        self._n_windows = n_windows

        total = n_windows * self.window_size
        flat = np.empty(total, dtype=int)

        w = 0
        for n in range(n_windows):
            flat[w:w+self.half] = I0[n*self.half:(n+1)*self.half]
            w += self.half
            flat[w:w+self.half] = I1[n*self.half:(n+1)*self.half]
            w += self.half

        return flat.tolist()

    def __iter__(self):
        if self._cached_flat is None:
            self._cached_flat = self._build_stream_for_epoch()
        for idx in self._cached_flat:
            yield int(idx)

    def __len__(self):
        if self._n_windows is None:
            K0 = len(self.majority_idx) // self.half
            K1 = len(self.expanded_minority) // self.half
            n_windows = min(K0, K1)
        else:
            n_windows = self._n_windows
        return n_windows * self.window_size

class BalancedSFTTrainer(SFTTrainer):
    def _prepare_dataset(self, dataset, *args, **kwargs):
        prepared = super()._prepare_dataset(dataset, *args, **kwargs)
        columns_to_remove = [c for c in ("prompt", "completion") if c in getattr(prepared, "column_names", [])]
        if columns_to_remove:
            prepared = prepared.remove_columns(columns_to_remove)
        return prepared

    def _get_train_sampler(self, train_dataset=None):
        if self.args.group_by_length:
            return super()._get_train_sampler(train_dataset)

        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        if not hasattr(train_dataset, "column_names"):
            return super()._get_train_sampler(train_dataset)

        if "MOR_label" in train_dataset.column_names:
            labels = train_dataset["MOR_label"]
        elif "SepsisLabel" in train_dataset.column_names:
            labels = train_dataset["SepsisLabel"]
        else:
            return super()._get_train_sampler(train_dataset)

        world_size = getattr(self.accelerator, "num_processes", 1)
        micro_bs = self.args.per_device_train_batch_size
        grad_accum = self.args.gradient_accumulation_steps

        window_size = world_size * micro_bs * grad_accum

        if window_size % 2 != 0:
            window_size -= 1
        window_size = max(window_size, 2)

        return BalancedStreamSampler(
            labels=labels,
            window_size=window_size,
            minority_multiplier=1,
            seed=self.args.seed,
        )

class DropColumnsDataCollator:
    def __init__(self, base_collator, drop_columns):
        self.base_collator = base_collator
        self.drop_columns = set(drop_columns)

    def __call__(self, features):
        if self.drop_columns:
            features = [
                {k: v for k, v in feature.items() if k not in self.drop_columns}
                for feature in features
            ]
        return self.base_collator(features)



def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Script args: {script_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Model args: {model_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = datasets.load_dataset(
        "json",
        data_files=script_args.dataset_path,
        split="train",
    )
    
    training_args.remove_unused_columns = False

    logger.info(f"Loaded dataset with {len(dataset)} samples")
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template. "
            "assistant_only_loss path requires a chat tokenizer."
        )
    
    # Change chat template for Qwen3
    tmpl = tokenizer.chat_template
    tmpl = tmpl.replace(
        "<think>\\n\\n</think>\\n\\n", ""
    )
    tmpl = tmpl.replace(
        "{%- if loop.last or (not loop.last and reasoning_content) %}",
        "{%- if loop.last and reasoning_content %}",
    )
    tokenizer.chat_template = tmpl

    def truncate_dataset_left(dataset, max_length, map_kwargs=None):
        if map_kwargs is None:
            map_kwargs = {}
        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[-max_length:] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(
            truncate,
            batched=True,
            **map_kwargs,
        )
        return dataset

    trl_utils.truncate_dataset = truncate_dataset_left
    trl_sft_trainer.truncate_dataset = truncate_dataset_left

    trainer = BalancedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.data_collator = DropColumnsDataCollator(
        trainer.data_collator,
        drop_columns={"prompt", "completion"},
    )

    logger.info("*** Train ***")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
