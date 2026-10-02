# TRIAGE SFT

Supervised fine-tuning of `Qwen/Qwen3-4B-Base` on the dialectical-reasoning data, and the preprocessing that produces that data for all five benchmarks. The RL stage that starts from the resulting checkpoint is [`../triage_rl/`](../triage_rl/). What the model is and how to use a finished checkpoint are in the [top-level README](../README.md).

```
triage_sft/
├── data/
│   ├── P12/
│   │   ├── process_script/         # initial preprocessing (raw -> npy) + dataset description
│   │   ├── splits/                 # the five cross-validation index files (no patient data)
│   │   └── triage_process_script/  # textualize -> batch requests -> SFT data
│   ├── P19/
│   │   ├── process_script/
│   │   ├── splits/
│   │   └── triage_process_script/
│   ├── mimic3/
│   │   ├── process_script/
│   │   ├── splits/                 # readme only: one fixed split, five training seeds
│   │   └── triage_process_script/
│   ├── mimic4/
│   │   ├── process_script/
│   │   ├── splits/
│   │   └── triage_process_script/
│   └── eicu/
│       ├── process_script/
│       ├── splits/
│       └── triage_process_script/
├── recipes/
│   ├── accelerate_configs/
│   │   └── zero3.yaml              # DeepSpeed ZeRO-3 accelerate config
│   ├── p12_triage_sft_split1.yaml
│   ├── p19_triage_sft_split1.yaml
│   ├── mimic3_triage_sft_seed42.yaml
│   ├── mimic4_triage_sft_split1.yaml
│   └── eicu_triage_sft_split1.yaml
├── utils.py                        # model / tokenizer / wandb helpers
├── configs.py                      # ScriptArguments + SFTConfig
└── sft.py                          # SFT entry point
```

## 1. Environment

We run on a single H200 GPU.

Base image

```
docker.io/ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5
```

Additional packages (inside the container)

```bash
pip install transformers==4.57.3 accelerate==1.7.0 trl==0.25.1 deepspeed==0.18.2
pip install -U bitsandbytes
```

## 2. Data

Each dataset has a `data/<name>/process_script/` folder (initial preprocessing and a dataset description) and a `data/<name>/triage_process_script/` folder (textualization and SFT-data generation). See the per-dataset readmes: P12 ([process](data/P12/process_script/readme.md), [triage](data/P12/triage_process_script/readme.md)), P19 ([process](data/P19/process_script/readme.md), [triage](data/P19/triage_process_script/readme.md)), MIMIC-III ([process](data/mimic3/process_script/readme.md), [triage](data/mimic3/triage_process_script/readme.md)), MIMIC-IV ([process](data/mimic4/process_script/readme.md), [triage](data/mimic4/triage_process_script/readme.md)) and eICU ([process](data/eicu/process_script/readme.md), [triage](data/eicu/triage_process_script/readme.md)).

SFT data format. `sft.py` consumes a JSON file (path set via `dataset_path` in the recipe) where each example has:

- `prompt`: a chat-style list, e.g. `[{"role": "user", "content": "..."}]`.
- `completion`: a chat-style list, e.g. `[{"role": "assistant", "content": "..."}]`, holding the dialectical rationales and the final 0/1 answer.
- `MOR_label` (P12, MIMIC-III, MIMIC-IV, eICU) or `SepsisLabel` (P19): an `int` label (`0`/`1`) used by the class-balanced sampler.

## 3. Training

After preparing the dataset JSON:

```bash
export HF_TOKEN=xxxx
export WANDB_API_KEY=xxxx   # optional, for Weights & Biases logging

accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    sft.py \
    --config recipes/p12_triage_sft_split1.yaml
```

[`recipes/p12_triage_sft_split1.yaml`](recipes/p12_triage_sft_split1.yaml) holds the P12 SFT config. The split shown is only an example: within a dataset the hyperparameters are the same for every split, so just point `dataset_path` at the split you want and set `hub_model_id`. The other datasets have their own recipes: [`recipes/p19_triage_sft_split1.yaml`](recipes/p19_triage_sft_split1.yaml), [`recipes/mimic3_triage_sft_seed42.yaml`](recipes/mimic3_triage_sft_seed42.yaml) (MIMIC-III has one fixed split from KEDGN, so its recipe is named by the training seed instead), [`recipes/mimic4_triage_sft_split1.yaml`](recipes/mimic4_triage_sft_split1.yaml) and [`recipes/eicu_triage_sft_split1.yaml`](recipes/eicu_triage_sft_split1.yaml). MIMIC-IV and eICU use a longer `max_length` than the other three, because their sequences are longer.

`hub_model_id` is commented out in every recipe and has to be set before the run starts, because that is where the intermediate checkpoints go. `save_steps: 0.0833` writes a checkpoint twelve times over the three epochs, about every quarter epoch, while `save_total_limit: 1` keeps only the newest of them on disk. The earlier ones survive as commits in the Hub repo, which is what `push_to_hub: true` with `hub_strategy: every_save` is for. The repo is created private (`hub_private_repo: true`).

Which checkpoint the RL stage starts from. The recipes do not evaluate during training (`do_eval: false`), so validation AUPRC is not something the trainer reports. It is computed afterwards, by scoring the validation split with the two-stage procedure described in the [top-level README](../README.md) and reading the resulting per-record probabilities. The reported runs start from the checkpoint with the best validation AUPRC among epochs up to 2.0, which is not always the best checkpoint overall.
