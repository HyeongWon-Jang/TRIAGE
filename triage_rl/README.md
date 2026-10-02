# TRIAGE RL post-training

GRPO post-training of the SFT policy, built on [verl](https://github.com/volcengine/verl). Starting from a supervised fine-tuned checkpoint, the policy is optimised against a batch separation reward: a squared-hinge margin that pushes each positive patient's risk score above the negatives' within the rollout batch.

This directory is not a copy of verl. It holds the TRIAGE-specific files only: the reward, the sampler, the trainer changes that compute the batch-level reward, the data preparation, and the launch scripts. Everything else is upstream verl and is not duplicated here.

```
triage_rl/
├── verl_overlay/           files to copy into a verl checkout, at the paths shown
│   ├── verl/utils/reward_score/triage.py        the reward (new)
│   ├── verl/utils/reward_score/triage_p19.py    P19 wording of the same reward (new)
│   ├── verl/utils/reward_score/__init__.py      dispatch for the two data_source keys
│   ├── verl/experimental/dataset/sampler.py     the paired-patient samplers
│   ├── verl/experimental/reward_loop/reward_manager/naive.py
│   ├── verl/trainer/ppo/ray_trainer.py          the sliding-batch separation reward
│   ├── verl/trainer/config/reward/reward.yaml   the keys those read
│   ├── verl/utils/dataset/rl_dataset.py         row-wise overlong-prompt filtering
│   ├── verl/utils/vllm/vllm_fp32_logits_utils.py   float32 LM-head logits (new)
│   ├── verl/utils/net_utils.py                  free-port search over a fixed range
│   ├── verl/trainer/config/ppo_trainer.yaml     the root hydra config
│   ├── scripts/install_vllm_sglang_mcore.sh     verl's installer, one step appended
│   └── examples/data_preprocess/TRIAGE_dataset.py   JSON -> parquet
├── data_preprocess/        preparation for the long-prompt datasets
│   ├── filter_train_by_tokens.py
│   └── expand_test_swap.py
└── exp/                    one launch script per benchmark
    ├── P12_split1.sh  P19_split1.sh  MIMIC3_split1.sh
    └── eICU_split1.sh  MIMIC4_split1.sh
```

## 1. Environment

The reported runs were built on verl at commit [`c2345689`](https://github.com/volcengine/verl/tree/c2345689c894dc1f30f536a912da9bfe0d61b940) (which reports itself as 0.8.0.dev), with vLLM as the rollout engine. Start from that commit:

```bash
git clone https://github.com/volcengine/verl.git
cd verl && git checkout c2345689c894dc1f30f536a912da9bfe0d61b940
```

Copy the overlay over the checkout **before installing**, because the installer is one of the files it replaces:

```bash
cp -r /path/to/triage_rl/verl_overlay/* .
```

Then install, following verl's own [installation guide](https://verl.readthedocs.io/en/latest/start/install.html):

```bash
conda create -n verl python==3.12
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

`scripts/install_vllm_sglang_mcore.sh` is verl's installer with a final step appended. Step 2 constrains `numpy<2.0.0`; the appended step re-pins it to `numpy==2.2.6` and adds `math-verify`, `gpustat`, `scikit-learn` and `multiprocess==0.70.11`. Running verl's unmodified installer leaves the environment on a numpy the rest of the stack disagrees with, which is why this file is in the overlay rather than left to the checkout.

Five files have no counterpart upstream: `reward_score/triage.py`, `reward_score/triage_p19.py`, `utils/vllm/vllm_fp32_logits_utils.py`, `examples/data_preprocess/TRIAGE_dataset.py` and `reward_manager/gdpo.py`. The other 22 are copies of verl's own source with our changes. Every file carries a notice at the top saying which of the two it is. The substantive ones: `reward_score/__init__.py` dispatches the `triage` / `triage_p19` data sources; `experimental/dataset/sampler.py` replaces a 40-line abstract-interface stub with the two samplers below; `trainer/ppo/ray_trainer.py` carries the batch-level reward, the decision-token masking and the class-balance advantage scaling; `experimental/agent_loop/` produces the constrained second-stage logprobs the reward reads; `workers/actor/dp_actor.py` adds the decision-token cross-entropy term; `trainer/main_ppo.py` and `trainer/config/data/legacy_data.yaml` instantiate and configure the samplers; `reward_manager/gdpo.py` is the manager the launch scripts select; `utils/dataset/rl_dataset.py` reads `extra_info.split` per row when applying the prompt-length cap, so the cap lands on the train rows and leaves validation to the rollout's own check.

Three more are there because the rest will not import or run without them. `trainer/config/ppo_trainer.yaml` is the root hydra config `main_ppo.py` loads by name. `utils/vllm/vllm_fp32_logits_utils.py` can promote the LM-head matmul to float32, which raises logit resolution where bfloat16 is coarse; it is gated on `rollout.fp32_logits`, and every launch script here sets that to `False`, so the reported runs did not use it. `utils/net_utils.py` gives `get_free_port` a `port_range` argument, which backs `vllm_master_port_range`. The sibling setting `worker_master_port_range` needs no overlay file of its own: `ray_trainer.py` forwards it to verl's `RayWorkerGroup`, which already accepts a `master_port_range` upstream.

Copying these over a different verl version overwrites that version's changes to the same files. Diff before you copy if your checkout is not `c2345689`.

## 2. Data

Each split needs a train and a test parquet. The trainer reads `prompt` and the binary `ground_truth`, and dispatches the reward on `data_source` (`triage`, or `triage_p19` for P19).

```bash
python examples/data_preprocess/TRIAGE_dataset.py \
    --local_dir /path/to/triage_rl/exp/data \
    --train_data_source p12_split1_train.json \
    --train_output_parquet_name p12_split1_train.parquet \
    --test_data_source  p12_split1_validation.json \
    --test_output_parquet_name p12_split1_test.parquet
```

`--local_dir` has to be the launch scripts' own `data/` directory: `exp/*.sh` read `${SCRIPT_DIR}/data/<name>.parquet`, so the files must end up in `triage_rl/exp/data/`. The default (`./data`, relative to the verl checkout you run this from) is not that directory.

Each input JSON record needs `prompt` (a one-element chat list) and `MOR_label` (`0` survival / `1` in-hospital death; `SepsisLabel` for P19). P19's class imbalance is severe enough that its train set is the one we oversample, each minority stay repeated three times; the other four are used as built.

eICU and MIMIC-IV need two extra steps, both in `data_preprocess/`, because their prompts are about twice as long and each patient contributes a *pair* of rows:

- `filter_train_by_tokens.py` drops train stays whose longer variant exceeds the prompt cap. It keys on the stay, not the row, so it can never drop half a pair and leave an orphan, which the sampler asserts against. This is optional: every launch script sets `filter_overlong_prompts=True`, so the trainer applies the same cap row-wise on the train split when the parquet is loaded. Running it first moves the work earlier and makes the drop count visible (`report`) before a GPU is involved. The two are not identical, though: this script drops a stay on the longer of its two rows, while the loader filters rows independently, so a pair whose two lengths straddle the cap would lose one row and trip the sampler's orphan check. The two variants of a stay came out the same length everywhere we measured, which is why the loader path is usable at all.
- `expand_test_swap.py` expands the test set so each stay appears as a survival-first and a death-first row, using the same transform the training augmentation applies. It rewrites the survival / in-hospital-death wording only, so it is a no-op on P19, whose prompts are phrased around sepsis.

The invariants the parquets have to satisfy are: one `data_source` per file, both classes present, two rows per patient with matching labels, one survival-first and one death-first row per pair, and prompt token lengths within the configured cap.

The pairing applies to every dataset, not only the two that need `expand_test_swap.py`, and it is a prerequisite on the input rather than something the code arranges. The in-training validation aggregates by `extra_info.index // 2`, so rows `2k` and `2k+1` of the test JSON have to be the same patient under the two opposite outcome orderings; the converter assigns those indices in file order and nothing downstream checks that the pair belongs to one patient. The textualizers here emit one row per patient, so whatever builds your test JSON has to do the pairing, as the authors' inputs did. That validation is for watching a run in progress: the numbers the paper reports come from scoring saved checkpoints with a separate pipeline, so an unpaired test parquet distorts the curve rather than a reported result. On the train side the sampler raises on an orphan; nothing checks the test side before launch.

## 3. Training

Run these from the verl checkout. The scripts live in this directory, not in the checkout, so give the full path:

```bash
SFT_CHECKPOINT=/path/to/p12_split1_sft   bash /path/to/triage_rl/exp/P12_split1.sh
SFT_CHECKPOINT=/path/to/eicu_split1_sft  bash /path/to/triage_rl/exp/eICU_split1.sh
```

Set `HF_TOKEN` and `WANDB_API_KEY` first; each script aborts if either is unset. `SFT_CHECKPOINT` is the policy RL starts from. The reported runs start from the SFT checkpoint selected by validation AUPRC among epochs up to 2.0, which is not always the best SFT checkpoint overall. Those intermediate checkpoints live in the SFT run's Hub repo rather than on disk; the [SFT README](../triage_sft/README.md) explains where they come from and how the selection is made.

`PREFLIGHT=1` resolves the configuration and exits without touching a GPU.

What the run optimises. The advantage estimator is verl's GDPO with a single reward key, `separation_reward`, which reduces to per-group GRPO on that reward; the per-group standard-deviation division is disabled (Dr. GRPO). The actor loss adds a cross-entropy term on the decision token (`ce_loss_coef=0.25`) and a KL term. The sampled answer token itself is masked out of the policy loss, because the reward already depends on the decision through the constrained second-stage logprobs.

The reward (`triage.py`). A response must carry exactly the three required headers, in either rationale order; a malformed one short-circuits to the clip floor. Otherwise the two-class logprobs of `"0"` and `"1"` at the decision position are renormalised, and the per-sample reward is the clipped log-likelihood of the true class. The trainer then builds the batch-level separation reward from those per-sample scores. The trainer supports pooling recent rollout batches for the comparison, but every configuration here sets `sliding_pool_size=0`, so the ranking is within the current rollout batch only. Samples whose format failed carry no ranking signal and are floored to the group minimum minus its standard deviation, which is no penalty at all when the group's valid rewards are all equal, a case that does occur.

The samplers (`sampler.py`). `RolloutClassRatioSampler` draws each rollout batch at the natural class ratio, and `ClassBalancedMiniBatchSampler` at a fixed one. Both treat a patient's two reasoning-order rows as a unit; both raise rather than proceed if a pair is incomplete. Two passes per epoch are used so that both rows of each patient are seen.

One caveat on `RolloutClassRatioSampler`: its `__len__` is derived from the minority pool, but iteration stops when either pool runs out. Where rounding the class ratio up makes the majority pool bind first, the reported epoch length overstates what is actually drawn, and a few per cent of minority patients are not seen in that pass. The run is bounded by `total_training_steps` rather than by epochs, so this shifts which patients a given pass covers rather than how long training lasts.

## 4. Datasets

There is one script per benchmark and, apart from paths and names, they differ only in the prompt caps: 10240 / 16384 for P12 and MIMIC-III, 8192 / 10240 for P19, 16384 / 24576 for eICU and MIMIC-IV. Everything else matches, including two GPUs with 2-way Ulysses sequence parallelism. P19's minority oversampling happens in the SFT data, not here. Each script covers split 1; point the paths at another split to run it.
