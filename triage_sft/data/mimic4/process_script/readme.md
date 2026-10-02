# MIMIC-IV

## Dataset

MIMIC-IV is a single-center critical care database distributed via PhysioNet under credentialed access (https://physionet.org/content/mimiciv/1.0/). We use version 1.0 for in-hospital mortality prediction from the first 48 hours of recorded measurements, with the admission (`hadm_id`) as the unit of analysis. Following the GSNF preprocessing, the processed dataset has 26,070 admissions with 3,491 positive cases (13.39 %) and 96 clinical variables (intake and output, laboratory results and medication prescriptions), with no static features. See the paper's appendix for the full description.

## Preprocessing

For MIMIC-IV, we follow the preprocessing of [GSNF](https://github.com/mzgaooo/GSNF), so that our numbers and the baselines' are computed on the same cohort as the prior work we compare against.

The preprocessing produces the per-admission time series (`mimic4_full_dataset.csv`), the mortality labels (`mortality_labels.csv`) and the variable names (`tables/variable_name_dict.csv`), which we save under `processed_data/`.

Our code reads those through a per-stay store, `PTdict_list.npy`: one entry per admission with `arr` (the values), `mask`, `time` and `length`, alongside `arr_outcomes.npy` for the labels. The mask is the observation indicator the source data already carries.

## Train/val/test split

Five splits of 20,856 / 2,607 / 2,607, generated the same way as for eICU. The training seed stays fixed at 42 and the split is what varies.

The index files are under [`splits/`](../splits), so the exact splits behind the reported numbers can be reused rather than regenerated. Each is a permutation of `0..N-1` partitioned into train / validation / test, and carries no patient data.

MIMIC-IV is credentialed-access data (PhysioNet), so the raw and processed files are not redistributed here; only the split index files above are. Run the preprocessing yourself to generate the rest.
