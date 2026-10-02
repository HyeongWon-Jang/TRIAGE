# MIMIC-IV SFT-data generation

This documents how the MIMIC-IV (in-hospital mortality) SFT dataset is built. The procedure is the same as eICU's; see [`../../eicu/triage_process_script/readme.md`](../../eicu/triage_process_script/readme.md) for the stage-by-stage description. The MIMIC-IV-specific points are below.

Prerequisite: run the MIMIC-IV preprocessing in [`../process_script`](../process_script) first, which produces `mimic4_full_dataset.csv`, `mortality_labels.csv` and `tables/variable_name_dict.csv` under `../process_script/processed_data`.

## Stage 1: textualize

```
python textualize.py --processed_data_dir ../process_script/processed_data --output_path ./textualized_data.json
```

Same feature-centric rendering as eICU, with two differences:

- 96 variables, not 14. The feature list gives each variable a short description and its unit, and `variable_name_dict.csv` maps the variables to the data columns.
- Records are keyed by `file_name = mimic4_<hadm_id>`, the admission being the unit of analysis.

The question block is the same in-hospital-mortality prompt as MIMIC-III's and eICU's.

## Stage 2: build batch requests

```
python build_batch_requests.py \
    --textualized_path ./textualized_data.json \
    --output_dir ./requests \
    --minority_repeats 3 \
    --num_parts 6
```

Same procedure as eICU's; we split the requests into six parts, MIMIC-IV being roughly twice eICU's size.

## Stage 3: run the batch

As for MIMIC-III and eICU, the batch is run locally, because MIMIC-IV is credentialed-access data (PhysioNet) and cannot be sent to an external API. We serve Kimi K2 Thinking with vLLM (an OpenAI-compatible endpoint), send the stage 2 requests to it, and save one response per `custom_id`.

## Building the SFT dataset

Identical to eICU's assembly: pair each admission's two outcome rationales by sample index, stitch them into one completion, and add the swap-order augmented copy. Each example is `{"MOR_label", "prompt", "completion"}`.

The result is `data/mimic4/mimic4_triage_sft_split{N}.json`, one file per split, consumed by [`recipes/mimic4_triage_sft_split1.yaml`](../../../recipes/mimic4_triage_sft_split1.yaml).

Sequence lengths are the practical constraint here, as with eICU, which is why the recipe's `max_length` is set higher than for P12, P19 and MIMIC-III.
