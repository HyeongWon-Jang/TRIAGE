# MIMIC-III splits

MIMIC-III uses KEDGN's fixed split and training seeds 42 to 46, so this folder holds no index files.

MIMIC-III comes with one fixed train / validation / test split, produced by the KEDGN preprocessing as `mimic3_{train,val,test}_{x,y}.npy` under [`../process_script/processed_data`](../process_script). The partition is already baked into those arrays, so there is no index file to include, unlike the other four benchmarks.

The repetition protocol differs accordingly. P12, P19, eICU and MIMIC-IV vary the split across five folds and hold the training seed at 42; MIMIC-III holds the split and varies the training seed instead, over 42 to 46, and the dispersion reported for it is over those five runs. The convention is implemented in `baselines/strats/src/utils.py` (`seed_for`).
