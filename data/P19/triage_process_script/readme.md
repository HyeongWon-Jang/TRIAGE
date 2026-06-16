# P19 SFT-data generation

This documents how the PhysioNet 2019 (P19, sepsis early prediction) SFT dataset is built. We elicit outcome-specific ("dialectical") rationales from a strong LLM (GPT-5.1), separately for each candidate outcome, and assemble them into prompt/completion pairs.

Prerequisite: run the P19 preprocessing in [`../process_script`](../process_script) first (our modified version of the standard P19/Raindrop preprocessing). It produces the cleaned arrays (`PT_dict_list_6_cleaned.npy`, `arr_outcomes_6_cleaned.npy`, with `labels_ts.npy` and `labels_demogr.npy`) and the train/val/test splits (`phy19_split{N}_cleaned.npy`), used when aggregating each split. Stage 1 below reads those processed files.

The two scripts here cover stage 1 and stage 2. Stage 3 (running the batch) and the final assembly are described below without extra scripts.

## Stage 1: textualize

`textualize.py` renders each patient as a chat prompt (feature list, demographics, and per-feature (Time, Value) series) and attaches the `SepsisLabel`. It reads the processed P19 arrays (`PT_dict_list_6_cleaned.npy` and friends) from `../process_script/processed_data`.

```
python textualize.py --processed_data_dir ../process_script/processed_data --output_path ./textualized_data.json
```

The prompt it writes is the dialectical prompt the model is trained to answer: it asks for a rationale for sepsis, a rationale for no sepsis, and a final 0/1 decision.

## Stage 2: build batch requests

`build_batch_requests.py` reads the textualized data and writes OpenAI Batch API request files (JSONL) plus a manifest, over the entire dataset at once (no train/val/test split here). Each request prompts the strong LLM to assume one outcome at a time (sepsis and no sepsis) and to list only the features that support that assumed outcome. Minority-label patients (`SepsisLabel=1`) are repeated `--minority_repeats` times (6 for P19), which is how the minority class is oversampled. Responses are keyed by patient id, so the per-split selection happens at assembly time.

```
python build_batch_requests.py \
    --textualized_path ./textualized_data.json \
    --output_dir ./requests \
    --model gpt-5.1 \
    --minority_repeats 6
```

Use `--num_parts N` to split the requests into N JSONL files (even, request-based chunks) if you want to submit them as separate batches.

## Stage 3: run the batch

Submit the request `.jsonl` from stage 2 to the OpenAI Batch API (`/v1/chat/completions`, `OPENAI_API_KEY` required) and collect the responses. Keep, per patient id, each outcome's rationale for every sample index. We do not ship a runner for this; any thin wrapper around the OpenAI Batch API works.

## Building the SFT dataset

For each split, we select that split's training patients by loading the split's training indices (`phy19_split{N}_cleaned.npy`, which index into the cleaned arrays), mapping each index to a patient id through `PT_dict_list_6_cleaned.npy`, and matching that id to the `file_name` in `textualized_data.json` (the batch responses are keyed by the same id). We then pair each patient's two outcome-specific rationales by sample index, and build the dataset to include the two example types below. The `sepsis` and `no sepsis` rationales are stitched into one completion, in the same format the model is trained to produce. Swap-order augmentation adds a second example per pair with the two rationales (and the prompt's reasoning order) swapped, so the model does not latch onto a fixed ordering.

Each example is `{"SepsisLabel", "prompt", "completion"}` (the prompt feature list and time series are abbreviated as `...` here):

Type 1, original order:

```json
{
  "SepsisLabel": 1,
  "prompt": [{"role": "user", "content": "...feature list...\n\n## Question\nWill the patient experience sepsis onset within the next 6 hours?\n\nReasoning by the following process:\n1. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n2. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n3. Make a final decision: '0' for no sepsis onset within the next 6 hours, '1' for sepsis onset within the next 6 hours.\n\n...answer format...\n\n## Feature of the patient\n..."}],
  "completion": [{"role": "assistant", "content": "## Rationale for sepsis\n...\n\n## Rationale for no sepsis\n...\n\n## Final Decision\n1"}]
}
```

Type 2, swap-augmented (sepsis/no-sepsis order swapped in both the prompt's reasoning steps and the completion):

```json
{
  "SepsisLabel": 1,
  "prompt": [{"role": "user", "content": "...feature list...\n\n## Question\nWill the patient experience sepsis onset within the next 6 hours?\n\nReasoning by the following process:\n1. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n2. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n3. Make a final decision: '0' for no sepsis onset within the next 6 hours, '1' for sepsis onset within the next 6 hours.\n\n...answer format...\n\n## Feature of the patient\n..."}],
  "completion": [{"role": "assistant", "content": "## Rationale for no sepsis\n...\n\n## Rationale for sepsis\n...\n\n## Final Decision\n1"}]
}
```

The result is `data/P19/p19_triage_sft_split{N}.json` (one file per split), consumed by the SFT recipe `recipes/p19_triage_sft_split1.yaml`.
