# P12 (PhysioNet 2012)

## Dataset

P12 is the PhysioNet Challenge 2012 dataset. The task is in-hospital mortality prediction from the first 48 hours of a patient's ICU stay. It contains 12,000 ICU records (challenge sets A, B, and C), each with up to 36 irregularly sampled clinical time-series variables and 5 static descriptors (age, gender, height, ICU type, weight); survival is the majority class. We start from the Raindrop/ViTST release and additionally remove 12 outlier patients (following Set Functions for Time Series), leaving 11,988. See Appendix C.1 of the paper for the full description, and PhysioNet (https://physionet.org/content/challenge-2012/1.0.0/) for the original data.

## Preprocessing

This folder contains our modified `ParseData.py` for the P12 dataset. The remaining preprocessing scripts (`IrregularSampling.py`, `removeoutliers.py`) are taken as-is from Raindrop and ViTST.

### Pipeline

The full P12 preprocessing pipeline runs the following scripts in order:

1. `ParseData.py` (modified, provided in this folder): parses the raw dataset.
2. `IrregularSampling.py` (unmodified): permutes and reorganizes the structure of the parsed data.
3. `removeoutliers.py` (unmodified): removes 12 outlier patients, following Set Functions for Time Series.
4. Train / val / test split (see below).

### Train/val/test split

We do not run `Generate_splitID.py` for P12. Instead, we reuse the official train/val/test split IDs released by [Raindrop](https://github.com/mims-harvard/Raindrop) and [ViTST](https://github.com/Leezekun/ViTST), which is the standard split used by prior work on P12 and ensures direct comparability with reported baselines.
