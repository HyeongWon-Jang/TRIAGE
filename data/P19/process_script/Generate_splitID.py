import numpy as np

# arr_outcomes = np.load('../processed_data/arr_outcomes.npy', allow_pickle=True)

# generate 5 random train/val/test splits
"""Use 8:1:1 split"""
p_train = 0.80
p_val   = 0.10
p_test  = 0.10


n =  38738  # original 38803 patients, remove 65 outliers
n_train = round(n*p_train)
n_val   = round(n*p_val)
n_test  = n - (n_train+n_val)
print(n_train, n_val, n_test)
Nsplits = 5
for j in range(Nsplits):
    rng = np.random.default_rng(42+j)
    p = rng.permutation(n)
    # p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val   = p[n_train:n_train+n_val]
    idx_test  = p[n_train+n_val:]
    
    split_ids = np.array([idx_train, idx_val, idx_test], dtype=object)
    np.save('../splits/phy19_split'+str(j+1)+'_cleaned.npy', split_ids)

print('split IDs saved')
