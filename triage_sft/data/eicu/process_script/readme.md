# eICU

## Dataset

eICU is the eICU Collaborative Research Database, a multi-center US critical care database distributed via PhysioNet under credentialed access (https://physionet.org/content/eicu-crd/2.0/). We use it for in-hospital mortality prediction from the first 48 hours of the ICU stay. Following the GSNF preprocessing, the processed dataset has 12,312 ICU stays with 2,168 positive cases (17.61 %) and 14 clinical variables, with no static features. See the paper's appendix for the full description.

## Preprocessing

For eICU, we follow the preprocessing of [GSNF](https://github.com/mzgaooo/GSNF), so that our numbers and the baselines' are computed on the same cohort as the prior work we compare against.

The preprocessing produces the per-stay time series (`eicu_data.csv`) and the mortality labels (`eicu_labels.csv`), which we save under `processed_data/`.

Our code reads those through a per-stay store, `PTdict_list.npy`: one entry per stay with `arr` (the values), `mask`, `time` and `length`, alongside `arr_outcomes.npy` for the labels. The mask is the observation indicator the source data already carries.

## Train/val/test split

Five splits of 9,849 / 1,231 / 1,232, following the GSNF preprocessing. The training seed stays fixed at 42 and the split is what varies, as for P12 and P19; MIMIC-III does the opposite.

The index files are under [`splits/`](../splits), so the exact splits behind the reported numbers can be reused rather than regenerated. Each is a permutation of `0..N-1` partitioned into train / validation / test, and carries no patient data.

eICU is credentialed-access data (PhysioNet), so the raw and processed files are not redistributed here; only the split index files above are. Run the preprocessing yourself to generate the rest.
