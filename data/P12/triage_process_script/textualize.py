import argparse
import json
import decimal
from pathlib import Path

import numpy as np
import pandas as pd

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

TS_PARAMS_ORDER = [
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "Mg",
    "MAP",
    "MechVent",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
]

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

BLACKLIST = {
    "140501",
    "150649",
    "140936",
    "143656",
    "141264",
    "145611",
    "142998",
    "147514",
    "142731",
    "150309",
    "155655",
    "156254",
}


def round_up(x, place=0):
    context = decimal.getcontext()
    original_rounding = context.rounding
    context.rounding = decimal.ROUND_CEILING
    rounded = round(decimal.Decimal(str(x)), place)
    context.rounding = original_rounding
    return float(rounded)


def _is_valid(value):
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False


def _is_missing(value):
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
        return value is not None and float(value) < 0
    except (TypeError, ValueError):
        return False


def construct_demogr_description(static_demogr):
    desc = []
    age, gender, height, icu_type, weight = static_demogr

    if _is_valid(age):
        desc.append(f"{int(float(age))} years old")

    try:
        gender_val = int(float(gender))
    except (TypeError, ValueError):
        gender_val = None

    if gender_val == 0:
        desc.append("female")
    elif gender_val == 1:
        desc.append("male")

    if _is_valid(height):
        desc.append(f"{float(height)} cm")

    if _is_valid(weight):
        desc.append(f"{float(weight)} kg")

    try:
        icu_val = int(float(icu_type))
    except (TypeError, ValueError):
        icu_val = None

    icu_map = {
        1: "coronary care unit",
        2: "cardiac surgery recovery unit",
        3: "medical ICU",
        4: "surgical ICU",
    }
    if icu_val in icu_map:
        desc.append(f"stayed in {icu_map[icu_val]}")

    if desc:
        return "A patient is " + ", ".join(desc) + "."
    return ""


def build_feature_centric_features(df_data):
    merged = {}
    for _, row in df_data.iterrows():
        param = row["param"]
        if param == "Weight":
            continue
        value = row["value"]
        if _is_missing(value):
            continue
        ts = row["time"]
        hrs, mins = float(ts[0:2]), float(ts[3:5])
        hour = hrs + (mins / 60.0)
        hour_key = round_up(hour, 1)

        if hour_key not in merged:
            merged[hour_key] = {}
        if param not in merged[hour_key]:
            merged[hour_key][param] = []
        merged[hour_key][param].append(round(float(value), 2) if isinstance(value, float) else value)

    feature_series = {feature: [] for feature in TS_PARAMS_ORDER}
    for hour_key in sorted(merged.keys()):
        for feature, values in merged[hour_key].items():
            if feature in feature_series:
                for value in values:
                    feature_series[feature].append((hour_key, value))

    lines = []
    lines.append("The patient's clinical features are organized in a feature-centric manner.")
    lines.append(
        "For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours since ICU admission.\n"
    )
    for feature in TS_PARAMS_ORDER:
        if not feature_series[feature]:
            continue
        pairs = ", ".join(f"({t}, {v})" for t, v in feature_series[feature])
        lines.append(f"### {feature}")
        lines.append(pairs + "\n")

    return "\n".join(lines).rstrip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="../rawdata",
        help="Directory containing Outcomes-*.txt and set-a/b/c/",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./textualized_data.json",
        help="Output JSON file with textualized features and prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_data_dir)
    output_path = Path(args.output_path)

    outcomes_a = pd.read_csv(
        raw_dir / "Outcomes-a.txt",
        sep=",",
        header=0,
        names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"],
    )
    outcomes_b = pd.read_csv(
        raw_dir / "Outcomes-b.txt",
        sep=",",
        header=0,
        names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"],
    )
    outcomes_c = pd.read_csv(
        raw_dir / "Outcomes-c.txt",
        sep=",",
        header=0,
        names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"],
    )
    outcomes = pd.concat([outcomes_a, outcomes_b, outcomes_c], axis=0, ignore_index=True)
    outcomes_map = dict(zip(outcomes["RecordID"].astype(str), outcomes["In-hospital_death"]))

    results = []
    set_dirs = [raw_dir / "set-a", raw_dir / "set-b", raw_dir / "set-c"]
    files = []
    for set_dir in set_dirs:
        files.extend(sorted(p for p in set_dir.iterdir() if p.suffix == ".txt"))
    total = len(files)

    for idx, path in enumerate(files, start=1):
        file_id = path.stem
        if file_id in BLACKLIST:
            print("Skipping blacklisted file:", file_id)
            continue
        df = pd.read_csv(path, sep=",", header=1, names=["time", "param", "value"])
        static_keys = ["Age", "Gender", "Height", "ICUType", "Weight"]
        df_demogr = df[df["param"].isin(static_keys)]
        df_data = df[~df["param"].isin(static_keys)]
        static_map = {}
        for _, row in df_demogr.iterrows():
            static_map.setdefault(row["param"], row["value"])
        static = (
            static_map.get("Age", -1),
            static_map.get("Gender", -1),
            static_map.get("Height", -1),
            static_map.get("ICUType", -1),
            static_map.get("Weight", -1),
        )

        demogr = construct_demogr_description(static)
        time_series = build_feature_centric_features(df_data)
        feature_block = "## Feature of the patient\n" + "\n".join(p for p in [demogr, time_series] if p)
        prompt = "\n\n".join(p for p in [FEATURE_LIST_P12, QUESTION_BLOCK, feature_block] if p)
        results.append(
            {
                "file_name": file_id,
                "patient_features": feature_block,
                "prompt": [{"role": "user", "content": prompt}],
                "MOR_label": int(outcomes_map.get(file_id, 0)),
            }
        )
        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
