import argparse
import json
import decimal
from pathlib import Path

import numpy as np

FEATURE_LIST_MIMIC3 = (
    "I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, "
    "each characterized by number of features.\n"
    "The list of features are as follows:\n\n"
    "- Time: Time of the measurement, in hours after admission\n"
    "- Weight: Weight, in kg\n"
    "- HR: Heart rate, in bpm\n"
    "- MBP: Mean blood pressure, in mmHg\n"
    "- DBP: Diastolic blood pressure, in mmHg\n"
    "- SBP: Systolic blood pressure, in mmHg\n"
    "- SpO2: Oxygen saturation, in percentage\n"
    "- RR: Respiratory rate, in breaths/min\n"
    "- CRR: Capillary refill rate\n"
    "- Glucose: Serum glucose, in mg/dL\n"
    "- pH: Blood pH, in pH\n"
    "- Temperature: Body temperature, in deg C\n"
    "- FiO2: Fraction of inspired oxygen (0-1)\n"
    "- GCS-EO: Eye opening score of Glasgow coma scale, in 4-point scale\n"
    "- GCS-MR: Motor response score of Glasgow coma scale, in 6-point scale\n"
    "- GCS-T: Total Glasgow coma scale score, in 15-point scale\n"
    "- GCS-VR: Verbal response score of Glasgow coma scale, in 5-point scale\n"
)

TS_PARAMS_ORDER = [
    "Weight",
    "Heart Rate",
    "Mean blood pressure",
    "Diastolic blood pressure",
    "Systolic blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Capillary refill rate",
    "Glucose",
    "pH",
    "Temperature",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
]

FEATURE_ALIAS = {
    "Weight": "Weight",
    "Heart Rate": "HR",
    "Mean blood pressure": "MBP",
    "Diastolic blood pressure": "DBP",
    "Systolic blood pressure": "SBP",
    "Oxygen saturation": "SpO2",
    "Respiratory rate": "RR",
    "Capillary refill rate": "CRR",
    "Glucose": "Glucose",
    "pH": "pH",
    "Temperature": "Temperature",
    "Fraction inspired oxygen": "FiO2",
    "Glascow coma scale eye opening": "GCS-EO",
    "Glascow coma scale motor response": "GCS-MR",
    "Glascow coma scale total": "GCS-T",
    "Glascow coma scale verbal response": "GCS-VR",
}

GCS_VALUE_TEXT = {
    "Glascow coma scale eye opening": {
        1: "No Response",
        2: "To Pain",
        3: "To Speech",
        4: "Spontaneously",
    },
    "Glascow coma scale motor response": {
        1: "No Response",
        2: "Abnormal Extension",
        3: "Abnormal Flexion",
        4: "Flex-withdraws",
        5: "Localizes Pain",
        6: "Obeys Commands",
    },
    "Glascow coma scale verbal response": {
        0: "No Response-ETT",
        1: "No Response",
        2: "Incomprehensible Sounds",
        3: "Inappropriate Words",
        4: "Confused",
        5: "Oriented",
    },
}

QUESTION_BLOCK = (
    "Based on the given feature of a patient, answer the question below.\n\n"
    "## Question\n"
    "Will the patient experience in-hospital death during this ICU stay?\n\n"
    "Reasoning by the following process:\n"
    "1. If the patient indeed survives, which of the patient's given features might be the cause?\n"
    "2. If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause?\n"
    "3. Make a final decision: '0' for survival, '1' for in-hospital death.\n\n"
    "Your answer format must be as follows:\n"
    "```\n"
    "## Rationale for survival\n"
    "[possible justification if patient survives]\n\n"
    "## Rationale for in-hospital death\n"
    "[possible justification if patient experiences in-hospital death]\n\n"
    "## Final Decision\n"
    "[0 (survival) or 1 (in-hospital death); respond by single number only]\n"
    "```\n"
)


def round_up(x, place=0):
    context = decimal.getcontext()
    original_rounding = context.rounding
    context.rounding = decimal.ROUND_CEILING
    rounded = round(decimal.Decimal(str(x)), place)
    context.rounding = original_rounding
    return float(rounded)


def _is_valid(value):
    try:
        return value is not None and np.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def construct_demogr_description(demo):
    if demo is None or len(demo) == 0:
        return ""
    height = float(demo[0])
    if _is_valid(height):
        if height <= 240.0:
            return f"A patient is {round(height, 2)} cm."
    return ""


def build_feature_centric_features(time_arr, values_arr, mask_arr, length):
    merged = {}
    valid_len = int(length)

    for t_idx in range(valid_len):
        hour = round_up(float(time_arr[t_idx]), 1)
        if hour not in merged:
            merged[hour] = {}

        for f_idx, feature in enumerate(TS_PARAMS_ORDER):
            is_observed = bool(mask_arr[t_idx, f_idx])
            value = values_arr[t_idx, f_idx]
            if not is_observed or not np.isfinite(value):
                continue
            if feature in GCS_VALUE_TEXT:
                code = int(round(float(value)))
                label = GCS_VALUE_TEXT[feature].get(code)
                disp_value = f"{code} {label}" if label is not None else code
            else:
                disp_value = round(float(value), 2)

            if feature not in merged[hour]:
                merged[hour][feature] = []
            merged[hour][feature].append(disp_value)

    feature_series = {feature: [] for feature in TS_PARAMS_ORDER}
    for hour in sorted(merged.keys()):
        for feature, value_list in merged[hour].items():
            for value in value_list:
                feature_series[feature].append((hour, value))

    lines = []
    lines.append("The patient's clinical features are organized in a feature-centric manner.")
    lines.append(
        "For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission.\n"
    )

    for feature in TS_PARAMS_ORDER:
        if not feature_series[feature]:
            continue
        pairs = ", ".join(f"({t}, {v})" for t, v in feature_series[feature])
        lines.append(f"### {FEATURE_ALIAS.get(feature, feature)}")
        lines.append(pairs + "\n")

    return "\n".join(lines).rstrip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="../process_script/processed_data",
        help="Directory containing mimic3_{train,val,test}_{x,y}.npy files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./textualized_data.json",
        help="Output JSON file with textualized features and prompts.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to include.",
    )
    parser.add_argument(
        "--limit_per_split",
        type=int,
        default=-1,
        help="Optional max number of samples per split. -1 means all.",
    )
    return parser.parse_args()


def _to_int_label(y_value):
    if isinstance(y_value, np.ndarray):
        if y_value.size == 0:
            return 0
        return int(y_value.reshape(-1)[0])
    return int(y_value)


def main():
    args = parse_args()
    processed_data_dir = Path(args.processed_data_dir)
    output_path = Path(args.output_path)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    results = []
    for split in split_names:
        x_path = processed_data_dir / f"mimic3_{split}_x.npy"
        y_path = processed_data_dir / f"mimic3_{split}_y.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing input files for split '{split}': {x_path}, {y_path}")

        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)

        if len(X) != len(y):
            raise ValueError(f"Length mismatch for split '{split}': len(X)={len(X)}, len(y)={len(y)}")

        max_n = len(X) if args.limit_per_split < 0 else min(args.limit_per_split, len(X))
        for idx in range(max_n):
            demo, time_arr, values_arr, mask_arr, length = X[idx]
            demogr = construct_demogr_description(demo)
            time_series = build_feature_centric_features(time_arr, values_arr, mask_arr, length)
            feature_block = "## Feature of the patient\n" + "\n".join(
                p for p in [demogr, time_series] if p
            )
            prompt = "\n\n".join(
                p for p in [FEATURE_LIST_MIMIC3, QUESTION_BLOCK, feature_block] if p
            )

            results.append(
                {
                    "file_name": f"{split}_{idx:05d}",
                    "split": split,
                    "patient_features": feature_block,
                    "prompt": [{"role": "user", "content": prompt}],
                    "MOR_label": _to_int_label(y[idx]),
                }
            )

        print(f"Processed split '{split}': {max_n}/{len(X)} records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
