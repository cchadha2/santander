import pandas as pd
import numpy as np

submission = pd.read_csv('../preds/lgbm_preds_1.4.csv')

set_1 = np.load('../output/public_LB.npy')
set_2 = np.load('../output/private_LB.npy')
synthetic_samples = np.load('../output/synthetic_samples_indexes.npy')

print(set_1)
print(set_2)
print(synthetic_samples)

submission.iloc[synthetic_samples, 1] = 0
submission.iloc[set_2, 1] = 0

print(submission)

submission.to_csv('../preds/set_1_test.csv', index=False)