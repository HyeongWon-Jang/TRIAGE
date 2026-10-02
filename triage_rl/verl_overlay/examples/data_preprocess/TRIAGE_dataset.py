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
"""Convert the TRIAGE prompt/label JSON files into the parquet format the trainer reads.

Each record needs a chat-style ``prompt`` and a binary label: ``MOR_label`` for P12,
MIMIC-III, eICU and MIMIC-IV, ``SepsisLabel`` for P19. The ``data_source`` written into
each row selects the reward module at training time.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
import sys

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_source", default="TRAIN", help="Train data source name from HuggingFace or local json file path")
    parser.add_argument("--test_data_source", default="TEST", help="Test data source name from HuggingFace or local json file path")
    parser.add_argument("--train_output_parquet_name", default="train.parquet", help="File name of output Train parquet file")
    parser.add_argument("--test_output_parquet_name", default="test.parquet", help="File name of output Test parquet file")
    parser.add_argument("--local_dir", default="./data",
                        help="directory the parquet files are written to")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset")

    args = parser.parse_args()

    train_data_source = args.train_data_source
    test_data_source = args.test_data_source
    if train_data_source.endswith(".json"):
        print(f"Loading the {train_data_source} dataset from local json file...", flush=True)
        train_dataset = datasets.load_dataset("json", data_files=train_data_source)["train"]
    else:
        print(f"Loading the {train_data_source} dataset from HuggingFace...", flush=True)
        train_dataset = datasets.load_dataset(train_data_source, trust_remote_code=True)["train"]

    if test_data_source.endswith(".json"):
        print(f"Loading the {test_data_source} dataset from local json file...", flush=True)
        test_dataset = datasets.load_dataset("json", data_files=test_data_source)["train"] # Here, if we load json file by this, we should set ["train"] to get the dataset.
    else:
        print(f"Loading the {test_data_source} dataset from HuggingFace...", flush=True)
        test_dataset = datasets.load_dataset(test_data_source, trust_remote_code=True)["test"]
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split, source):
        def process_fn(example, idx):
            prompt = example.pop('prompt')
            if 'MOR_label' in example:
                solution = example.pop('MOR_label')
            elif 'SepsisLabel' in example:
                solution = example.pop('SepsisLabel')
            else:
                raise KeyError("record has neither 'MOR_label' nor 'SepsisLabel'")
            assert solution in [0, 1]
            data = {
                "data_source": source,
                "prompt": prompt,
                "ability": "medical",
                "reward_model": {"style": "rule", "ground_truth": str(solution)}, # solution is 0 or 1, so we need to convert it to string.
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    # P19 is scored against the sepsis wording and every other dataset against the
    # in-hospital-death wording; data_source selects which. Getting it backwards does not
    # raise: every response fails the format check and the whole run sits at the reward
    # clip floor, learning nothing.
    tmp_data_source = "triage_p19" if "p19" in train_data_source.lower() else "triage"
    # Assume: If "p19" in train_data_source, "p19" also in test_data_source
    train_dataset = train_dataset.map(function=make_map_fn("train", tmp_data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", tmp_data_source), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir
    
    # Shuffle the dataset with a fixed seed
    train_dataset = train_dataset.shuffle(seed=args.seed)
    test_dataset = test_dataset.shuffle(seed=args.seed)

    train_dataset.to_parquet(os.path.join(local_dir, args.train_output_parquet_name))
    test_dataset.to_parquet(os.path.join(local_dir, args.test_output_parquet_name))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
