# MIMIC-III SFT-data generation

This documents how the MIMIC-III (in-hospital mortality) SFT dataset is built. We elicit outcome-specific ("dialectical") rationales from a strong LLM (Kimi K2 Thinking), separately for each candidate outcome, and assemble them into prompt/completion pairs.

Prerequisite: run the MIMIC-III preprocessing in [`../process_script`](../process_script) first. We follow the KEDGN preprocessing (built on the SeFT `medical_ts_datasets` pipeline) exactly, which produces `mimic3_{train,val,test}_{x,y}.npy` under `../process_script/processed_data`. We additionally use the Height value from that preprocessing as an extra static feature. MIMIC-III has a single fixed train/val/test split given by the KEDGN preprocessing (not several splits like P12/P19).

The two scripts here cover stage 1 and stage 2. Stage 3 (running the batch) and the final assembly are described below without extra scripts.

## Stage 1: textualize

`textualize.py` renders each patient as a chat prompt (feature list, the Height-based demographic line, and per-feature (Time, Value) series) and attaches the `MOR_label`. It reads the KEDGN `mimic3_{train,val,test}_{x,y}.npy` arrays.

```
python textualize.py --processed_data_dir ../process_script/processed_data --output_path ./textualized_data.json
```

The prompt it writes is the dialectical prompt the model is trained to answer: it asks for a rationale for survival, a rationale for in-hospital death, and a final 0/1 decision.

## Stage 2: build batch requests

`build_batch_requests.py` reads the textualized data and writes batch request files (JSONL) plus a manifest, over the entire dataset at once. Each request prompts the strong LLM to assume one outcome at a time (survival and in-hospital death) and to list only the features that support that assumed outcome. Minority-label patients are repeated `--minority_repeats` times (default 3 for MIMIC-III), which is how the minority class is oversampled.

```
python build_batch_requests.py \
    --textualized_path ./textualized_data.json \
    --output_dir ./requests \
    --minority_repeats 3
```

Use `--num_parts N` to split the requests into N JSONL files (even, request-based chunks) if you want to run them as separate batches.

## Stage 3: run the batch

Unlike P12 and P19, the MIMIC-III batch is run locally, because MIMIC-III is credentialed-access data (PhysioNet) and cannot be sent to an external API. We serve Kimi K2 Thinking with SGLang (an OpenAI-compatible endpoint), send the stage 2 requests to it, and save one response per `custom_id`.

## Building the SFT dataset

Here the split is already recorded in `textualized_data.json`: each record carries a `split` field (`train`/`val`/`test`) and a `file_name` like `train_00042`, so we select the `train` records directly (no separate index mapping). We gather each patient's responses from the id-keyed pool, pair the two outcome-specific rationales by sample index, and build the dataset to include the two example types below. The `survival` and `in-hospital death` rationales are stitched into one completion, in the same format the model is trained to produce. Swap-order augmentation adds a second example per pair with the two rationales (and the prompt's reasoning order) swapped, so the model does not latch onto a fixed ordering.

Each example is `{"MOR_label", "prompt", "completion"}` (the prompt feature list and time series are abbreviated as `...` here):

Type 1, original order:

```json
{
  "MOR_label": 1,
  "prompt": [{"role": "user", "content": "...feature list...\n\n## Question\nWill the patient experience in-hospital death during this ICU stay?\n\nReasoning by the following process:\n1. If the patient indeed survives, which of the patient's given features might be the cause?\n2. If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause?\n3. Make a final decision: '0' for survival, '1' for in-hospital death.\n\n...answer format...\n\n## Feature of the patient\n..."}],
  "completion": [{"role": "assistant", "content": "## Rationale for survival\n...\n\n## Rationale for in-hospital death\n...\n\n## Final Decision\n1"}]
}
```

Type 2, swap-augmented (survival/death order swapped in both the prompt's reasoning steps and the completion):

```json
{
  "MOR_label": 1,
  "prompt": [{"role": "user", "content": "...feature list...\n\n## Question\nWill the patient experience in-hospital death during this ICU stay?\n\nReasoning by the following process:\n1. If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause?\n2. If the patient indeed survives, which of the patient's given features might be the cause?\n3. Make a final decision: '0' for survival, '1' for in-hospital death.\n\n...answer format...\n\n## Feature of the patient\n..."}],
  "completion": [{"role": "assistant", "content": "## Rationale for in-hospital death\n...\n\n## Rationale for survival\n...\n\n## Final Decision\n1"}]
}
```

MIMIC-III uses the single train split given by KEDGN, so the result is a single file `data/mimic3/mimic3_triage_sft.json`, consumed by the SFT recipe `recipes/mimic3_triage_sft_seed42.yaml` (named by its training seed).
