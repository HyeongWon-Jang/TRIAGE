# eICU SFT-data generation

This documents how the eICU (in-hospital mortality) SFT dataset is built. As for MIMIC-III, we elicit outcome-specific ("dialectical") rationales from a strong LLM (Kimi K2 Thinking), separately for each candidate outcome, and assemble them into prompt/completion pairs.

Prerequisite: run the eICU preprocessing in [`../process_script`](../process_script) first, which produces `eicu_data.csv` and `eicu_labels.csv` under `../process_script/processed_data`. eICU has no static features, so unlike MIMIC-III the prompt has no demographic line. eICU has five train/val/test splits (see [`../process_script`](../process_script)); they are applied only when the SFT dataset is assembled.

The two scripts here cover stage 1 and stage 2. Stage 3 (running the batch) and the final assembly are described below without extra scripts.

## Stage 1: textualize

`textualize.py` renders each ICU stay as a chat prompt (feature list and per-feature (Time, Value) series, with Time in hours after ICU admission) and attaches the `MOR_label`. Each record is keyed by `file_name = eicu_<stay_id>`, so no split is baked in at this stage.

```
python textualize.py --processed_data_dir ../process_script/processed_data --output_path ./textualized_data.json
```

The prompt it writes is the same dialectical prompt as MIMIC-III's: it asks for a rationale for survival, a rationale for in-hospital death, and a final 0/1 decision.

## Stage 2: build batch requests

`build_batch_requests.py` is the MIMIC-III script with the eICU feature list. It reads the textualized data and writes batch request files (JSONL) plus a manifest, over the entire cohort at once. Each request prompts the strong LLM to assume one outcome at a time (survival and in-hospital death) and to list only the features that support that assumed outcome. Minority-label stays are repeated `--minority_repeats` times (3 for eICU), which is how the minority class is oversampled. We split the requests into four parts.

```
python build_batch_requests.py \
    --textualized_path ./textualized_data.json \
    --output_dir ./requests \
    --minority_repeats 3 \
    --num_parts 4
```

## Stage 3: run the batch

As for MIMIC-III, the eICU batch is run locally, because eICU is credentialed-access data (PhysioNet) and cannot be sent to an external API. We serve Kimi K2 Thinking with SGLang (an OpenAI-compatible endpoint), send the stage 2 requests to it, and save one response per `custom_id`.

## Building the SFT dataset

Responses are keyed by stay id, so a stay's rationales are generated once and reused by every split that puts it in its training set. For each split, we select that split's training stays by their `eicu_<stay_id>` file names, pair each stay's two outcome-specific rationales by sample index, and stitch them into one completion. Swap-order augmentation adds a second example per pair with the two rationales (and the prompt's reasoning order) swapped. The examples have the same form as MIMIC-III's (see its [readme](../../mimic3/triage_process_script/readme.md)): `{"MOR_label", "prompt", "completion"}`.

The result is `data/eicu/eicu_triage_sft_split{N}.json`, one file per split, consumed by [`recipes/eicu_triage_sft_split1.yaml`](../../../recipes/eicu_triage_sft_split1.yaml).
