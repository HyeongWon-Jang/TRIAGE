import argparse
import json
import decimal
from pathlib import Path

import pandas as pd

FEATURE_LIST_EICU = (
    "I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, "
    "each characterized by number of features.\n"
    "The list of features are as follows:\n\n"
    "- Time: Time of the measurement, in hours after admission\n"
    "- Glucose: Blood glucose, in mg/dL\n"
    "- DBP: Diastolic blood pressure, in mmHg\n"
    "- SBP: Systolic blood pressure, in mmHg\n"
    "- SpO2: Oxygen saturation, in percentage\n"
    "- RR: Respiratory rate, in breaths/min\n"
    "- GCS-MR: Motor response score of Glasgow coma scale, in 1-6 scale\n"
    "- GCS-EO: Eye opening score of Glasgow coma scale, in 1-4 scale\n"
    "- MAP: Mean arterial pressure, in mmHg\n"
    "- HR: Heart rate, in bpm\n"
    "- GCS-T: Total Glasgow coma scale score, in 3-15 scale\n"
    "- GCS-VR: Verbal response score of Glasgow coma scale, in 1-5 scale\n"
    "- pH: Blood pH, in pH\n"
    "- FiO2: Fraction of inspired oxygen, in percentage (0-100)\n"
    "- Temperature: Body temperature, in deg C\n"
)

# eICU variables, in the order of the Value_i / Mask_i columns of eicu_data.csv.
TS_PARAMS_ORDER = [
    "glucose",
    "Invasive BP Diastolic",
    "Invasive BP Systolic",
    "O2 Saturation",
    "Respiratory Rate",
    "Motor",
    "Eyes",
    "MAP (mmHg)",
    "Heart Rate",
    "GCS Total",
    "Verbal",
    "pH",
    "FiO2",
    "Temperature (C)",
]

FEATURE_ALIAS = {
    "glucose": "Glucose",
    "Invasive BP Diastolic": "DBP",
    "Invasive BP Systolic": "SBP",
    "O2 Saturation": "SpO2",
    "Respiratory Rate": "RR",
    "Motor": "GCS-MR",
    "Eyes": "GCS-EO",
    "MAP (mmHg)": "MAP",
    "Heart Rate": "HR",
    "GCS Total": "GCS-T",
    "Verbal": "GCS-VR",
    "pH": "pH",
    "FiO2": "FiO2",
    "Temperature (C)": "Temperature",
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


def build_feature_centric_features(time_arr, values_arr, mask_arr):
    # time_arr is in minutes after ICU admission.
    merged = {}

    for t_idx in range(len(time_arr)):
        hour = round_up(float(time_arr[t_idx]) / 60.0, 1)
        if hour not in merged:
            merged[hour] = {}

        for f_idx, feature in enumerate(TS_PARAMS_ORDER):
            if mask_arr[t_idx, f_idx] != 1:
                continue
            if feature not in merged[hour]:
                merged[hour][feature] = []
            merged[hour][feature].append(round(float(values_arr[t_idx, f_idx]), 2))

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
        help="Directory containing eicu_data.csv and eicu_labels.csv.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./textualized_data.json",
        help="Output JSON file with textualized features and prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional max number of ICU stays. -1 means all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_data_dir = Path(args.processed_data_dir)
    output_path = Path(args.output_path)

    df = pd.read_csv(processed_data_dir / "eicu_data.csv", index_col=0)
    labels = pd.read_csv(processed_data_dir / "eicu_labels.csv", index_col="ID")["labels"]

    value_cols = [f"Value_{i}" for i in range(len(TS_PARAMS_ORDER))]
    mask_cols = [f"Mask_{i}" for i in range(len(TS_PARAMS_ORDER))]

    # One record per ICU stay, keyed by stay id; the five splits are applied later,
    # when the SFT dataset is assembled.
    stay_ids = sorted(df.index.unique())
    if args.limit >= 0:
        stay_ids = stay_ids[: args.limit]
    groups = df.groupby(df.index)

    results = []
    for stay_id in stay_ids:
        rows = groups.get_group(stay_id).sort_values("Time")
        time_series = build_feature_centric_features(
            rows["Time"].to_numpy(), rows[value_cols].to_numpy(), rows[mask_cols].to_numpy()
        )
        # eICU has no static features, so there is no demographic line.
        feature_block = "## Feature of the patient\n" + time_series
        prompt = "\n\n".join([FEATURE_LIST_EICU, QUESTION_BLOCK, feature_block])

        results.append(
            {
                "file_name": f"eicu_{stay_id}",
                "stay_id": int(stay_id),
                "patient_features": feature_block,
                "prompt": [{"role": "user", "content": prompt}],
                "MOR_label": int(labels.loc[stay_id]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
