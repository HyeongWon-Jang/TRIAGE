import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUEST_TAG = "all"
MINORITY_LABEL = 1
OUTCOME_PHRASES = ("in-hospital death", "survival")

FEATURE_LIST_P12 = (
    "I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, "
    "each characterized by number of features.\n"
    "The list of features are as follows:\n\n"
    "- Time: Time of the measurement, in hours after admission\n"
    "- Albumin: Serum albumin, in g/dL\n"
    "- ALP: Alkaline phosphatase, in U/L\n"
    "- ALT: Alanine aminotransferase, in U/L\n"
    "- AST: Aspartate aminotransferase, in U/L\n"
    "- Bilirubin: Total bilirubin, in mg/dL\n"
    "- BUN: Blood urea nitrogen, in mg/dL\n"
    "- Cholesterol: Cholesterol, in mg/dL\n"
    "- Creatinine: Serum creatinine, in mg/dL\n"
    "- DiasABP: Diastolic blood pressure (Invasive), in mmHg\n"
    "- FiO2: Fractional inspired O2 (0-1)\n"
    "- GCS: Glasgow coma score (3-15)\n"
    "- Glucose: Serum glucose, in mg/dL\n"
    "- HCO3: Bicarbonate, in mmol/L\n"
    "- HCT: Hematocrit, in percentage\n"
    "- HR: Heart rate, in bpm\n"
    "- K: Serum potassium, in mmol/L\n"
    "- Lactate: Lactate, in mmol/L\n"
    "- Mg: Serum magnesium, in mmol/L\n"
    "- MAP: Mean arterial pressure (Invasive), in mmHg\n"
    "- MechVent: Mechanical ventilation\n"
    "- Na: Serum sodium, in mmol/L\n"
    "- NIDiasABP: Diastolic blood pressure (Non-invasive), in mmHg\n"
    "- NIMAP: Mean arterial pressure (Non-invasive), in mmHg\n"
    "- NISysABP: Systolic blood pressure (Non-invasive), in mmHg\n"
    "- PaCO2: Partial pressure of carbon dioxide, in mmHg\n"
    "- PaO2: Partial pressure of oxygen, in mmHg\n"
    "- pH: Arterial pH (0-14)\n"
    "- Platelets: Platelet count, in x10^9/L\n"
    "- RespRate: Respiration rate, in breaths/min\n"
    "- SaO2: Oxygen saturation, in percentage\n"
    "- SysABP: Systolic blood pressure (Invasive), in mmHg\n"
    "- Temp: Temperature (deg C)\n"
    "- TroponinI: Troponin-I, in ug/L\n"
    "- TroponinT: Troponin-T, in ng/mL\n"
    "- Urine: Urine output, in mL\n"
    "- WBC: White blood cell count, in x10^9/L\n"
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
        default=3,
        help="Number of repetitions for minority label samples.",
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


def _slug_outcome(outcome_phrase):
    return outcome_phrase.replace(" ", "_").replace("-", "_")


def build_labelwise_prompt(patient_features, outcome_phrase):
    question_block = (
        "Based on the given features of a patient, provide possible clinical rationale for why the patient could "
        f"{outcome_phrase} during this ICU stay.\n"
        f"Assume the outcome is {outcome_phrase} and list only supporting features.\n"
        "- Use only the provided features; do not invent data.\n"
        "- Do not describe normal or healthy findings as abnormal.\n"
        "- Do not describe abnormal findings as normal or healthy.\n"
        "- Do not mention any label, class, or numeric outcome.\n"
        "- Do not discuss the opposite outcome.\n"
        "- If there is no supporting evidence, leave the rationale blank.\n\n"
        "Your answer format must be:\n"
        "```\n"
        f"## Rationale for {outcome_phrase}\n"
        "[blank or concise bullet points]\n"
        "```\n"
    )
    return "\n\n".join([FEATURE_LIST_P12, question_block, patient_features])


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
        label = int(record.get("MOR_label", 0))
        repeats = args.minority_repeats if label == MINORITY_LABEL else 1
        for outcome_phrase in OUTCOME_PHRASES:
            prompt_messages = [
                {"role": "user", "content": build_labelwise_prompt(record["patient_features"], outcome_phrase)}
            ]
            outcome_key = _slug_outcome(outcome_phrase)
            for sample_idx in range(1, repeats + 1):
                custom_id = f"{REQUEST_TAG}-{file_name}-{outcome_key}-{sample_idx}"
                manifest[custom_id] = {
                    "file_name": file_name,
                    "label": label,
                    "outcome_phrase": outcome_phrase,
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
