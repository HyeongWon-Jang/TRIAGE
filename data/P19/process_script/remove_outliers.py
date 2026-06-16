import re

import numpy as np

"""Remove all-missing patients (no non-zero observations in 34 time-series vars)."""

PT_PATH = '../processed_data/PT_dict_list_6.npy'
OUTCOME_PATH = '../processed_data/arr_outcomes_6.npy'
PT_CLEAN_PATH = '../processed_data/PT_dict_list_6_cleaned.npy'
OUTCOME_CLEAN_PATH = '../processed_data/arr_outcomes_6_cleaned.npy'

pt_dict_list = np.load(PT_PATH, allow_pickle=True)
arr_outcomes = np.load(OUTCOME_PATH, allow_pickle=True)
if arr_outcomes.ndim == 1:
    arr_outcomes = arr_outcomes.reshape(-1, 1)
elif arr_outcomes.ndim != 2 or arr_outcomes.shape[1] != 1:
    raise ValueError(f'Unexpected arr_outcomes shape: {arr_outcomes.shape} (expected (N,1) or (N,))')

if len(pt_dict_list) != len(arr_outcomes):
    raise ValueError(
        f'Length mismatch: PT_dict_list_6={len(pt_dict_list)} vs arr_outcomes_6={len(arr_outcomes)}'
    )

remove_indices = []
remove_ids = []
for i, sample in enumerate(pt_dict_list):
    arr = np.asarray(sample['arr'], dtype=float)
    length = max(0, min(int(sample['length']), arr.shape[0]))
    values = arr[:length, :34]
    nonzero_count = int(((values != 0) & np.isfinite(values)).sum())
    if nonzero_count == 0:
        remove_indices.append(i)
        pid = str(sample.get('id', f'sample_{i:05d}'))
        m = re.search(r'(\d{6})(?=\.psv$)', pid.replace('\\', '/'))
        remove_ids.append(m.group(1) if m else pid)

print('Original:', len(pt_dict_list), arr_outcomes.shape)
print('Remove all-missing patients:', len(remove_indices))
if remove_indices:
    print('Example IDs:', remove_ids[:10])

# remove by index (same ordering for both arrays)
pt_dict_list = np.delete(pt_dict_list, remove_indices)
arr_outcomes = np.delete(arr_outcomes, remove_indices, axis=0)

print('After remove:', len(pt_dict_list), arr_outcomes.shape)

np.save(PT_CLEAN_PATH, pt_dict_list)
np.save(OUTCOME_CLEAN_PATH, arr_outcomes)
print('Saved cleaned PT:', PT_CLEAN_PATH)
print('Saved cleaned outcomes:', OUTCOME_CLEAN_PATH)
