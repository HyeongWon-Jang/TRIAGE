import argparse
import json
import decimal
from pathlib import Path

import pandas as pd

FEATURE_LIST_MIMIC4 = (
    "I will provide you with medical information from Intensive Care Unit (ICU) visit of a patient, "
    "each characterized by number of features.\n"
    "The list of features are as follows:\n\n"
    "- Time: Time of the measurement, in hours after admission\n"
    "- Potassium Chloride: Potassium chloride administered, in mEq\n"
    "- Magnesium Sulfate: Magnesium sulfate administered, in grams\n"
    "- Calcium Gluconate: Calcium gluconate administered, in grams\n"
    "- PO Intake: Oral fluid intake, in ml\n"
    "- Insulin - Glargine: Insulin glargine administered, in units\n"
    "- Insulin - Regular: Regular insulin administered, in units\n"
    "- LR: Lactated Ringer's solution administered, in ml\n"
    "- Furosemide (Lasix): Furosemide administered, in mg\n"
    "- OR Crystalloid Intake: Crystalloid fluid administered intraoperatively, in ml\n"
    "- OR Cell Saver Intake: Cell saver (salvaged blood) administered intraoperatively, in ml\n"
    "- Solution: Solution administered, in ml\n"
    "- Dextrose 5%: Dextrose 5% solution administered, in ml\n"
    "- Piggyback: Piggyback (secondary) infusion administered, in ml\n"
    "- Phenylephrine: Phenylephrine administered, in mg\n"
    "- KCL (Bolus): Potassium chloride bolus administered, in ml\n"
    "- Albumin 5%: Albumin 5% solution administered, in ml\n"
    "- PT: Prothrombin time, in sec\n"
    "- PTT: Partial thromboplastin time, in sec\n"
    "- Basophils: Basophils, differential percentage, in % (0-100)\n"
    "- Eosinophils: Eosinophils, differential percentage, in % (0-100)\n"
    "- Hematocrit: Hematocrit level, in % (0-100)\n"
    "- Hemoglobin: Blood hemoglobin level, in g/dL\n"
    "- Lymphocytes: Lymphocytes, differential percentage, in % (0-100)\n"
    "- MCH: Mean corpuscular hemoglobin, in pg\n"
    "- MCV: Mean corpuscular volume, in fL\n"
    "- Monocytes: Monocytes, differential percentage, in % (0-100)\n"
    "- Neutrophils: Neutrophils, differential percentage, in % (0-100)\n"
    "- RDW: Red cell distribution width, in % (0-100)\n"
    "- Red Blood Cells: Red blood cell count, in M/uL\n"
    "- White Blood Cells: White blood cell count, in K/uL\n"
    "- Anion Gap: Anion gap, in mEq/L\n"
    "- Chloride: Serum chloride, in mEq/L\n"
    "- Creatinine: Serum creatinine, in mg/dL\n"
    "- Magnesium: Serum magnesium, in mg/dL\n"
    "- Phosphate: Serum phosphate, in mg/dL\n"
    "- Potassium: Potassium level, in mEq/L\n"
    "- Urea Nitrogen: Blood urea nitrogen, in mg/dL\n"
    "- Base Excess: Base excess, in mEq/L\n"
    "- Calculated Total CO2: Total carbon dioxide (calculated), in mEq/L\n"
    "- pCO2: Partial pressure of carbon dioxide, in mm Hg\n"
    "- pO2: Partial pressure of oxygen, in mm Hg\n"
    "- Lactate: Lactate, in mmol/L\n"
    "- Platelet Count: Platelet count, in K/uL\n"
    "- pH: pH of blood, urine, or other body fluid, in pH\n"
    "- Bicarbonate: Bicarbonate, in mEq/L\n"
    "- Sodium: Serum sodium, in mEq/L\n"
    "- Specific Gravity: Urine specific gravity (dimensionless)\n"
    "- Glucose: Glucose level, in mg/dL\n"
    "- Foley: Urine output via Foley catheter, in ml\n"
    "- Chest Tube #1: Chest tube drainage output, in ml\n"
    "- OR Urine: Intraoperative urine output, in ml\n"
    "- Sodium Chloride 0.9% Flush (prescription): Prescribed sodium chloride 0.9% flush volume, in ml\n"
    "- Potassium Chloride (prescription): Prescribed potassium chloride dose, in mEq\n"
    "- Magnesium Sulfate (prescription): Prescribed magnesium sulfate dose, in grams\n"
    "- Acetaminophen (prescription): Prescribed acetaminophen dose, in mg\n"
    "- Docusate Sodium (prescription): Prescribed docusate sodium dose, in mg\n"
    "- Aspirin (prescription): Prescribed aspirin dose, in mg\n"
    "- Insulin (prescription): Prescribed insulin dose, in units\n"
    "- Metoprolol Tartrate (prescription): Prescribed metoprolol tartrate dose, in mg\n"
    "- Bisacodyl (prescription): Prescribed bisacodyl dose, in mg\n"
    "- Calcium, Total: Serum total calcium, in mg/dL\n"
    "- Void: Voided urine output, in ml\n"
    "- OR EBL: Estimated intraoperative blood loss, in ml\n"
    "- Emesis: Emesis (vomitus) output, in ml\n"
    "- Pantoprazole (prescription): Prescribed pantoprazole dose, in mg\n"
    "- Heparin (prescription): Prescribed heparin dose, in units\n"
    "- Lorazepam (Ativan): Lorazepam administered, in mg\n"
    "- Heparin Sodium: Heparin sodium administered, in units\n"
    "- Midazolam (Versed): Midazolam administered, in mg\n"
    "- Alanine Aminotransferase (ALT): Alanine aminotransferase, in IU/L\n"
    "- Alkaline Phosphatase: Alkaline phosphatase, in IU/L\n"
    "- Asparate Aminotransferase (AST): Aspartate aminotransferase, in IU/L\n"
    "- Bilirubin, Total: Total bilirubin, in mg/dL\n"
    "- Albumin: Serum albumin, in g/dL\n"
    "- Gastric Meds: Medication administered via gastric tube, in ml\n"
    "- GT Flush: Gastric tube (GT) flush administered, in ml\n"
    "- Norepinephrine: Norepinephrine administered, in mg\n"
    "- Pre-Admission: Pre-admission output volume, in ml\n"
    "- D5W (prescription): Prescribed dextrose 5% in water (D5W) volume, in ml\n"
    "- Metoprolol: Metoprolol administered, in mg\n"
    "- Packed Red Blood Cells: Packed red blood cells administered, in ml\n"
    "- Sterile Water: Sterile water administered, in ml\n"
    "- D5 1/2NS: Dextrose 5% in half-normal saline administered, in ml\n"
    "- Magnesium Sulfate (Bolus): Magnesium sulfate bolus administered, in ml\n"
    "- Oral Gastric: Orogastric tube output, in ml\n"
    "- Straight Cath: Straight catheterization urine output, in ml\n"
    "- K Phos: Potassium phosphate administered, in mmol\n"
    "- Morphine Sulfate: Morphine sulfate administered, in mg\n"
    "- Insulin - Humalog: Insulin lispro (Humalog) administered, in units\n"
    "- Nitroglycerin: Nitroglycerin administered, in mg\n"
    "- TF Residual: Tube feed (TF) residual volume, in ml\n"
    "- Jackson Pratt #1: Jackson-Pratt drain output, in ml\n"
    "- TF Residual Output: Tube feed (TF) residual output volume, in ml\n"
    "- Nasogastric: Nasogastric tube output, in ml\n"
    "- Stool: Stool output, in ml\n"
    "- Fecal Bag: Fecal bag output, in ml\n"
)

# Variable names as they appear in variable_name_dict.csv, in feature-list order.
TS_PARAMS_ORDER = [
    "Potassium Chloride",
    "Magnesium Sulfate",
    "Calcium Gluconate",
    "PO Intake",
    "Insulin - Glargine",
    "Insulin - Regular",
    "LR",
    "Furosemide (Lasix)",
    "OR Crystalloid Intake",
    "OR Cell Saver Intake",
    "Solution",
    "Dextrose 5%",
    "Piggyback",
    "Phenylephrine",
    "KCL (Bolus)",
    "Albumin 5%",
    "PT",
    "PTT",
    "Basophils",
    "Eosinophils",
    "Hematocrit",
    "Hemoglobin",
    "Lymphocytes",
    "MCH",
    "MCV",
    "Monocytes",
    "Neutrophils",
    "RDW",
    "Red Blood Cells",
    "White Blood Cells",
    "Anion Gap",
    "Chloride",
    "Creatinine",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Urea Nitrogen",
    "Base Excess",
    "Calculated Total CO2",
    "pCO2",
    "pO2",
    "Lactate",
    "Platelet Count",
    "pH",
    "Bicarbonate",
    "Sodium",
    "Specific Gravity",
    "Glucose",
    "Foley",
    "Chest Tube #1",
    "OR Urine",
    "Sodium Chloride 0.9%  Flush Drug",
    "Potassium Chloride Drug",
    "Magnesium Sulfate Drug",
    "Acetaminophen Drug",
    "Docusate Sodium Drug",
    "Aspirin Drug",
    "Insulin Drug",
    "Metoprolol Tartrate Drug",
    "Bisacodyl Drug",
    "Calcium, Total",
    "Void",
    "OR EBL",
    "Emesis",
    "Pantoprazole Drug",
    "Heparin Drug",
    "Lorazepam (Ativan)",
    "Heparin Sodium",
    "Midazolam (Versed)",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase",
    "Asparate Aminotransferase (AST)",
    "Bilirubin, Total",
    "Albumin",
    "Gastric Meds",
    "GT Flush",
    "Norepinephrine",
    "Pre-Admission",
    "D5W Drug",
    "Metoprolol",
    "Packed Red Blood Cells",
    "Sterile Water",
    "D5 1/2NS",
    "Magnesium Sulfate (Bolus)",
    "Oral Gastric",
    "Straight Cath",
    "K Phos",
    "Morphine Sulfate",
    "Insulin - Humalog",
    "Nitroglycerin",
    "TF Residual",
    "Jackson Pratt #1",
    "TF Residual Output",
    "Nasogastric",
    "Stool",
    "Fecal Bag",
]

FEATURE_ALIAS = {
    "Sodium Chloride 0.9%  Flush Drug": "Sodium Chloride 0.9% Flush (prescription)",
    "Potassium Chloride Drug": "Potassium Chloride (prescription)",
    "Magnesium Sulfate Drug": "Magnesium Sulfate (prescription)",
    "Acetaminophen Drug": "Acetaminophen (prescription)",
    "Docusate Sodium Drug": "Docusate Sodium (prescription)",
    "Aspirin Drug": "Aspirin (prescription)",
    "Insulin Drug": "Insulin (prescription)",
    "Metoprolol Tartrate Drug": "Metoprolol Tartrate (prescription)",
    "Bisacodyl Drug": "Bisacodyl (prescription)",
    "Pantoprazole Drug": "Pantoprazole (prescription)",
    "Heparin Drug": "Heparin (prescription)",
    "D5W Drug": "D5W (prescription)",
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
    # time_arr is in minutes from the admission's first recorded measurement.
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
        "For each feature, measurements are listed as (Time, Value) pairs in chronological order, where Time denotes hours after admission.\n"
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
        help="Directory containing mimic4_full_dataset.csv, mortality_labels.csv and tables/variable_name_dict.csv.",
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
        help="Optional max number of admissions. -1 means all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_data_dir = Path(args.processed_data_dir)
    output_path = Path(args.output_path)

    df = pd.read_csv(processed_data_dir / "mimic4_full_dataset.csv", index_col=0)
    labels = pd.read_csv(processed_data_dir / "mortality_labels.csv", index_col="ID")["labels"]
    name_dict = pd.read_csv(processed_data_dir / "tables" / "variable_name_dict.csv", index_col=0)

    label_code = dict(zip(name_dict["label"], name_dict["label_code"]))
    codes = [label_code[feature] for feature in TS_PARAMS_ORDER]
    value_cols = [f"Value_{c}" for c in codes]
    mask_cols = [f"Mask_{c}" for c in codes]

    # One record per admission, keyed by hadm_id; the five splits are applied later,
    # when the SFT dataset is assembled.
    hadm_ids = sorted(df.index.unique())
    if args.limit >= 0:
        hadm_ids = hadm_ids[: args.limit]
    groups = df.groupby(df.index)

    results = []
    for hadm_id in hadm_ids:
        rows = groups.get_group(hadm_id).sort_values("Time")
        time_series = build_feature_centric_features(
            rows["Time"].to_numpy(), rows[value_cols].to_numpy(), rows[mask_cols].to_numpy()
        )
        # The processed MIMIC-IV data has no static features, so there is no demographic line.
        feature_block = "## Feature of the patient\n" + time_series
        prompt = "\n\n".join([FEATURE_LIST_MIMIC4, QUESTION_BLOCK, feature_block])

        results.append(
            {
                "file_name": f"mimic4_{hadm_id}",
                "hadm_id": int(hadm_id),
                "patient_features": feature_block,
                "prompt": [{"role": "user", "content": prompt}],
                "MOR_label": int(labels.loc[hadm_id]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    print(f"Saved {len(results)} records to {output_path}")


if __name__ == "__main__":
    main()
