# TRIAGE

**Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs**

[![Models](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](https://huggingface.co/Hyeongwon)

> 🚧 **Code will be released soon.** Model checkpoints are already available on the [Hugging Face Hub](https://huggingface.co/Hyeongwon).

## Overview

TRIAGE casts clinical risk prediction over irregularly sampled time series as a reasoning task for large language models. At a high level, rather than emitting a bare risk score, the model reasons *dialectically*, weighing competing evidence for and against patient deterioration, and produces a risk estimate together with a human-readable rationale.

Full method details, training code, and the data-processing pipeline will accompany the code release.

## Models

All checkpoints are fine-tuned from [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) and released under CC BY-NC 4.0.

| Model | Data | Prediction task | Training |
|---|---|---|---|
| [TRIAGE-4B-P12-SFT](https://huggingface.co/Hyeongwon/TRIAGE-4B-P12-SFT) | P12 | In-hospital mortality | SFT |
| [TRIAGE-4B-P12-SFT-RL](https://huggingface.co/Hyeongwon/TRIAGE-4B-P12-SFT-RL) | P12 | In-hospital mortality | SFT + RL |
| [TRIAGE-4B-P19-SFT](https://huggingface.co/Hyeongwon/TRIAGE-4B-P19-SFT) | P19 | Sepsis early prediction | SFT |
| [TRIAGE-4B-P19-SFT-RL](https://huggingface.co/Hyeongwon/TRIAGE-4B-P19-SFT-RL) | P19 | Sepsis early prediction | SFT + RL |

**Repository layout.** Each model holds five cross-validation splits in separate `split_N/` subfolders (`split_1` ... `split_5`). For the `SFT-RL` models, per-split RL checkpoints were selected by validation AUPRC, and the SFT warm-start used to initialize RL is kept on each repo's `rl_init` branch.

### Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

split = "split_1"  # one of split_1 ... split_5
repo  = "Hyeongwon/TRIAGE-4B-P19-SFT-RL"

tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=split)
model     = AutoModelForCausalLM.from_pretrained(repo, subfolder=split, device_map="auto")
```

The models expect a task-specific input/output template; the full inference pipeline will ship with the code.

## Data

| Data | Source | Processed (Raindrop, CC BY 4.0) |
|---|---|---|
| **P12** | [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/), in-hospital mortality | [figshare](https://doi.org/10.6084/m9.figshare.19514341.v1) |
| **P19** | [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), sepsis early prediction | [figshare](https://doi.org/10.6084/m9.figshare.19514338.v1) |

**MIMIC-III.** A separate, access-controlled release of checkpoints based on MIMIC-III is under consideration, owing to the dataset's credentialed-access license (MIMIC-III derivatives cannot be redistributed openly).

Preprocessing and split-construction details will be documented with the code release.

## License

Model checkpoints are released under **CC BY-NC 4.0** (non-commercial). Datasets remain under their respective licenses; see the links above.

## Citation

A citation will be added with the code release.
