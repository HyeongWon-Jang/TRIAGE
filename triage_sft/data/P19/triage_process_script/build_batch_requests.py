import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUEST_TAG = "all"
MINORITY_LABEL = 1

FEATURE_LIST_P19 = (
    "I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, "
    "each characterized by number of features.\n"
    "The list of features are as follows:\n\n"
    "- Time: Time of the measurement, in hours\n"
    "- HR: Heart rate, in beats per minute (bpm)\n"
    "- O2Sat: Pulse oximetry oxygen saturation, in percent (%)\n"
    "- Temp: Body temperature, in degrees Celsius (deg C)\n"
    "- SBP: Systolic blood pressure, in mm Hg\n"
    "- MAP: Mean arterial pressure, in mm Hg\n"
    "- DBP: Diastolic blood pressure, in mm Hg\n"
    "- Resp: Respiratory rate, in breaths per minute\n"
    "- EtCO2: End-tidal carbon dioxide, in mm Hg\n"
    "- BaseExcess: Base excess (excess bicarbonate), in mmol/L\n"
    "- HCO3: Bicarbonate, in mmol/L\n"
    "- FiO2: Fractional inspired O2 (0-1)\n"
    "- pH: Arterial pH (0-14)\n"
    "- PaCO2: Arterial partial pressure of carbon dioxide, in mm Hg\n"
    "- SaO2: Arterial oxygen saturation, in percent (%)\n"
    "- AST: Aspartate transaminase, in IU/L\n"
    "- BUN: Blood urea nitrogen, in mg/dL\n"
    "- Alkalinephos: Alkaline phosphatase, in IU/L\n"
    "- Calcium: Serum calcium, in mg/dL\n"
    "- Chloride: Serum chloride, in mmol/L\n"
    "- Creatinine: Serum creatinine, in mg/dL\n"
    "- Bilirubin_direct: Direct bilirubin, in mg/dL\n"
    "- Glucose: Serum glucose, in mg/dL\n"
    "- Lactate: Lactic acid, in mg/dL\n"
    "- Magnesium: Serum magnesium, in mg/dL\n"
    "- Phosphate: Serum phosphate, in mg/dL\n"
    "- Potassium: Serum potassium, in mmol/L\n"
    "- Bilirubin_total: Total bilirubin, in mg/dL\n"
    "- TroponinI: Troponin I, in ng/mL\n"
    "- Hct: Hematocrit, in percent (%)\n"
    "- Hgb: Hemoglobin, in g/dL\n"
    "- PTT: Partial thromboplastin time, in seconds\n"
    "- WBC: White blood cell count, in count x10^3/uL\n"
    "- Fibrinogen: Fibrinogen, in mg/dL\n"
    "- Platelets: Platelet count, in count x10^3/uL\n"
)

OUTCOME_CONFIGS = (
    {
        "key": "sepsis",
        "assumption": "sepsis onset within the next 6 hours",
        "header": "Rationale for sepsis",
    },
    {
        "key": "no_sepsis",
        "assumption": "no sepsis onset within the next 6 hours",
        "header": "Rationale for no sepsis",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--textualized_path",
        type=str,
        default="./textualized_data.json",
        help="Path to textualized data JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./requests",
        help="Directory for batch jsonl/manifest files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model name for batch requests.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8000,
        help="Max completion tokens for each response.",
    )
    parser.add_argument(
        "--minority_repeats",
        type=int,
        default=6,
        help="Number of repetitions for minority label samples (SepsisLabel=1).",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=1,
        help="Split the request list into this many parts (request-based, even chunks).",
    )
    return parser.parse_args()


def build_request(custom_id, prompt_messages, model, max_completion_tokens):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": prompt_messages,
            "max_completion_tokens": max_completion_tokens,
            "store": False,
        },
    }


def build_labelwise_prompt(patient_features, outcome_cfg):
    assumption = outcome_cfg["assumption"]
    header = outcome_cfg["header"]

    question_block = (
        "Based on the given features of a patient, provide possible clinical rationale for why the patient could "
        f"experience {assumption}.\n"
        f"Assume the outcome is '{assumption}' and list only supporting features.\n"
        "- Use only the provided features; do not invent data.\n"
        "- Do not describe normal or healthy findings as abnormal.\n"
        "- Do not describe abnormal findings as normal or healthy.\n"
        "- Do not mention any label, class, or numeric outcome.\n"
        "- Do not discuss the opposite outcome.\n"
        "- If there is no supporting evidence, leave the rationale blank.\n\n"
        "Your answer format must be:\n"
        "```\n"
        f"## {header}\n"
        "[blank or concise bullet points]\n"
        "```\n"
    )
    return "\n\n".join([FEATURE_LIST_P19, question_block, patient_features])


def main():
    args = parse_args()

    textualized_path = Path(args.textualized_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with textualized_path.open() as f:
        data = json.load(f)

    # Build one flat list of requests over the entire dataset (no train/val/test
    # split here). Minority-label patients are repeated so the minority class is
    # oversampled. Responses are keyed by patient id, so each split is selected
    # later, when the SFT dataset is assembled.
    requests = []
    manifest = {}
    for record in data:
        if not record.get("patient_features"):
            continue
        file_name = str(record["file_name"])
        label = int(record.get("SepsisLabel", 0))
        repeats = args.minority_repeats if label == MINORITY_LABEL else 1
        for outcome_cfg in OUTCOME_CONFIGS:
            outcome_key = outcome_cfg["key"]
            prompt_messages = [
                {"role": "user", "content": build_labelwise_prompt(record["patient_features"], outcome_cfg)}
            ]
            for sample_idx in range(1, repeats + 1):
                custom_id = f"{REQUEST_TAG}-{file_name}-{outcome_key}-{sample_idx}"
                manifest[custom_id] = {
                    "file_name": file_name,
                    "SepsisLabel": label,
                    "target_outcome": outcome_cfg["assumption"],
                    "target_outcome_key": outcome_key,
                    "sample_index": sample_idx,
                    "prompt": prompt_messages,
                }
                requests.append(
                    build_request(
                        custom_id=custom_id,
                        prompt_messages=prompt_messages,
                        model=args.model,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                )

    if not requests:
        print("No requests to write.")
        return

    # Split the flat request list into num_parts even chunks (request-based).
    num_parts = max(1, args.num_parts)
    per_part = (len(requests) + num_parts - 1) // num_parts
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for part_idx in range(num_parts):
        part_requests = requests[part_idx * per_part : (part_idx + 1) * per_part]
        if not part_requests:
            continue
        part_manifest = {req["custom_id"]: manifest[req["custom_id"]] for req in part_requests}

        input_path = output_dir / f"batch_input_{REQUEST_TAG}_part{part_idx + 1}of{num_parts}_{ts}.jsonl"
        with input_path.open("w", encoding="utf-8") as f:
            for req in part_requests:
                f.write(json.dumps(req, ensure_ascii=True))
                f.write("\n")

        manifest_path = output_dir / f"batch_manifest_{REQUEST_TAG}_part{part_idx + 1}of{num_parts}_{ts}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(part_manifest, f, ensure_ascii=True, indent=2)

        print(f"part {part_idx + 1}/{num_parts}: {len(part_requests)} requests -> {input_path.name}")

    print(f"Wrote {len(requests)} requests across {num_parts} part(s) to {output_dir}.")


if __name__ == "__main__":
    main()
