import argparse
import decimal
import json
import re
from pathlib import Path

import numpy as np

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

QUESTION_BLOCK = (
    "Based on the given feature of a patient, answer the question below.\n\n"
    "## Question\n"
    "Will the patient experience sepsis onset within the next 6 hours?\n\n"
    "Reasoning by the following process:\n"
    "1. If the patient indeed experiences sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n"
    "2. If the patient indeed does not experience sepsis onset within the next 6 hours, which of the patient's given features might be the cause?\n"
    "3. Make a final decision: '0' for no sepsis onset within the next 6 hours, '1' for sepsis onset within the next 6 hours.\n\n"
    "Your answer format must be as follows:\n"
    "```\n"
    "## Rationale for sepsis\n"
    "[possible justification if patient experiences sepsis onset within the next 6 hours]\n\n"
    "## Rationale for no sepsis\n"
    "[possible justification if patient does not experience sepsis onset within the next 6 hours]\n\n"
    "## Final Decision\n"
    "[0 (for no sepsis onset within the next 6 hours) or 1 (for sepsis onset within the next 6 hours); respond by single number only]\n"
    "```\n"
)

STATIC_ORDER = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]


def round_up(x, place=0):
    context = decimal.getcontext()
    original_rounding = context.rounding
    context.rounding = decimal.ROUND_CEILING
    rounded = round(decimal.Decimal(str(x)), place)
    context.rounding = original_rounding
    return float(rounded)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_static(value):
    val = _as_float(value)
    if val is None:
        return True
    if not np.isfinite(val):
        return True
    return val == -1.0


def _extract_short_file_name(original_id, fallback_idx):
    raw = str(original_id) if original_id is not None else f"sample_{fallback_idx:05d}"
    normalized = raw.replace("\\", "/")
    match = re.search(r"(\d{6})(?=\.psv$)", normalized)
    if match:
        return match.group(1), raw
    stem = Path(normalized).stem
    digits = re.search(r"(\d{6})$", stem)
    if digits:
        return digits.group(1), raw
    return stem or f"sample_{fallback_idx:05d}", raw


def construct_demogr_description(static_values):
    if static_values is None or len(static_values) != 6:
        return ""

    age, gender, unit1, unit2, hosp_adm_time, iculos = static_values
    age_text = ""
    gender_text = ""
    unit_text = ""
    hosp_text = ""
    iculos_text = ""

    if not _is_missing_static(age):
        age_v = float(age)
        age_text = f"{int(round(age_v))} years old"

    if not _is_missing_static(gender):
        gender_v = int(round(float(gender)))
        if gender_v == 0:
            gender_text = "female"
        elif gender_v == 1:
            gender_text = "male"

    unit1_flag = (not _is_missing_static(unit1)) and int(round(float(unit1))) == 1
    unit2_flag = (not _is_missing_static(unit2)) and int(round(float(unit2))) == 1
    if unit1_flag and not unit2_flag:
        unit_text = "medical ICU"
    elif unit2_flag and not unit1_flag:
        unit_text = "surgical ICU"

    if not _is_missing_static(hosp_adm_time):
        hadm = float(hosp_adm_time)
        if hadm < 0:
            hosp_text = (
                f"ICU admission occurred {round(abs(hadm), 2)} hours after hospital admission"
            )
        elif hadm > 0:
            hosp_text = (
                f"ICU admission occurred {round(hadm, 2)} hours before hospital admission"
            )
        else:
            hosp_text = "ICU and hospital admission were recorded at the same time"

    if not _is_missing_static(iculos):
        iculos_text = (
            f"the observations were recorded about {round(float(iculos), 2)} hours after ICU admission"
        )

    parts = []
    first = []
    if age_text:
        first.append(age_text)
    if gender_text:
        first.append(gender_text)
    if unit_text:
        first.append(f"went to {unit_text}")
    if first:
        parts.append("A patient is " + ", ".join(first) + ".")

    second = []
    if hosp_text:
        second.append(hosp_text)
    if iculos_text:
        second.append(iculos_text)
    if second:
        parts.append(", ".join(second) + ".")

    if parts:
        return " ".join(parts)
    return ""


def build_feature_centric_features(sample, ts_params_order):
    arr = np.asarray(sample["arr"], dtype=float)
    time_arr = np.asarray(sample["time"], dtype=float).reshape(-1)
    length = int(sample["length"])

    merged = {}
    for t_idx in range(max(0, min(length, arr.shape[0]))):
        hour_raw = float(time_arr[t_idx])
        # Keep hour=0 records; only invalid negative timestamps are skipped.
        if hour_raw < 0:
            continue
        hour = round_up(hour_raw, 1)
        if hour not in merged:
            merged[hour] = {}

        for f_idx, feature in enumerate(ts_params_order):
            value = arr[t_idx, f_idx]
            if not np.isfinite(value):
                continue
            # Zero values are missing placeholders.
            if value == 0:
                continue
            merged[hour].setdefault(feature, []).append(round(float(value), 2))

    feature_series = {feature: [] for feature in ts_params_order}
    for hour in sorted(merged.keys()):
        for feature, values in merged[hour].items():
            for value in values:
                feature_series[feature].append((hour, value))

    lines = []
    lines.append("The patient's clinical features are organized in a feature-centric manner.")
    lines.append(
        "For each feature, measurements are listed as (Time, Value) pairs in chronological order.\n"
    )

    for feature in ts_params_order:
        if not feature_series[feature]:
            continue
        pairs = ", ".join(f"({t}, {v})" for t, v in feature_series[feature])
        lines.append(f"### {feature}")
        lines.append(pairs + "\n")

    return "\n".join(lines).rstrip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="../process_script/processed_data",
        help="Directory containing PT_dict_list_6_cleaned.npy and related files.",
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
    processed_data_dir = Path(args.processed_data_dir)
    output_path = Path(args.output_path)

    ptdict_path = processed_data_dir / "PT_dict_list_6_cleaned.npy"
    outcomes_path = processed_data_dir / "arr_outcomes_6_cleaned.npy"
    labels_ts_path = processed_data_dir / "labels_ts.npy"
    labels_demogr_path = processed_data_dir / "labels_demogr.npy"

    pt_dict_list = np.load(ptdict_path, allow_pickle=True)
    outcomes = np.load(outcomes_path, allow_pickle=True).reshape(-1)
    labels_ts = np.load(labels_ts_path, allow_pickle=True).tolist()
    labels_demogr = np.load(labels_demogr_path, allow_pickle=True).tolist()

    if labels_ts and labels_ts[-1] == "SepsisLabel":
        ts_params_order = labels_ts[:-1]
    else:
        ts_params_order = labels_ts

    if labels_demogr != STATIC_ORDER:
        print(f"Warning: unexpected demographic order: {labels_demogr}")
    if len(ts_params_order) != 34:
        raise ValueError(f"Expected 34 time-series variables, got {len(ts_params_order)}")
    if len(pt_dict_list) != len(outcomes):
        raise ValueError(
            f"Length mismatch: PT_dict_list_6_cleaned={len(pt_dict_list)} vs arr_outcomes_6_cleaned={len(outcomes)}"
        )

    results = []
    for idx in range(len(pt_dict_list)):
        sample = pt_dict_list[idx]
        label = int(float(outcomes[idx]))
        short_name, original_name = _extract_short_file_name(sample.get("id"), idx)

        demogr = construct_demogr_description(sample.get("extended_static"))
        time_series = build_feature_centric_features(sample, ts_params_order)
        feature_block = "## Feature of the patient\n" + "\n".join(
            part for part in [demogr, time_series] if part
        )
        prompt = "\n\n".join(part for part in [FEATURE_LIST_P19, QUESTION_BLOCK, feature_block] if part)

        results.append(
            {
                "file_name": short_name,
                "original_file_name": original_name,
                "patient_features": feature_block,
                "prompt": [{"role": "user", "content": prompt}],
                "SepsisLabel": label,
            }
        )

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(pt_dict_list):
            print(f"Processed {idx + 1}/{len(pt_dict_list)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
