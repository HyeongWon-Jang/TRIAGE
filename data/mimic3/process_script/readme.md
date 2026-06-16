# MIMIC-III

## Dataset

MIMIC-III is a publicly released critical care database distributed via PhysioNet under credentialed access (https://physionet.org/content/mimiciii/1.4/). We use it for in-hospital mortality prediction. Following the SeFT and KEDGN preprocessing, the processed dataset has 21,107 samples with 16 clinical variables. See Appendix C.1 of the paper for the full description.

## Preprocessing

For MIMIC-III, we follow the preprocessing presented in [KEDGN](https://github.com/easonLuo2001/KEDGN) exactly, which builds on the [SeFT](https://github.com/ExpectationMax/medical_ts_datasets) `medical_ts_datasets` pipeline. We do not modify it.

The KEDGN preprocessing produces the train / val / test arrays, which we save under `processed_data/` as:

```
processed_data/mimic3_train_x.npy
processed_data/mimic3_train_y.npy
processed_data/mimic3_val_x.npy
processed_data/mimic3_val_y.npy
processed_data/mimic3_test_x.npy
processed_data/mimic3_test_y.npy
```

In addition, we use the Height information provided by that preprocessing as an extra static feature for each patient.

MIMIC-III is credentialed-access data (PhysioNet), so the raw and processed files are not redistributed here. Run the KEDGN preprocessing yourself to generate them.
