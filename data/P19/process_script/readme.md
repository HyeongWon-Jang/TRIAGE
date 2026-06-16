# P19 (PhysioNet 2019)

## Dataset

P19 is the PhysioNet / Computing in Cardiology Challenge 2019 dataset for early prediction of sepsis. The task is to predict whether a patient will develop sepsis within the next 6 hours, from hourly ICU measurements. It covers 38,803 patients with 34 irregularly sampled clinical time-series variables plus demographic descriptors, and sepsis-positive cases are a small minority. After removing 65 patients with no valid observations, 38,738 remain. See Appendix C.1 of the paper for the full description, and PhysioNet (https://physionet.org/content/challenge-2019/1.0.0/) for the original data.

## Preprocessing

For P19, no custom preprocessing scripts are provided in this folder. The preprocessing pipeline applies only standard, unmodified scripts from the [Raindrop](https://github.com/mims-harvard/Raindrop) preprocessing code.

### Pipeline

1. `remove_outliers.py`: filters out patients with no valid observations (65 removed, leaving 38,738).
2. `Generate_splitID.py`: generates fresh train / val / test IDs in an 8 : 1 : 1 ratio.

### Train/val/test split

Unlike P12, where we reuse the official Raindrop/ViTST split IDs, for P19 we generate a new split by running `Generate_splitID.py` with the default 8 : 1 : 1 proportion. Because this split is newly generated, we provide the resulting split files (`phy19_split{N}_cleaned.npy`) under [`splits/`](../splits) so the exact split we used can be reproduced.
