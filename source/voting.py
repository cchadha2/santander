import pandas as pd
import numpy as np

sub_df = pd.read_csv('../data/sample_submission.csv')

column_shuffle = pd.read_csv('../preds/lgbm_preds_shuffling_2.csv')
xgboost = pd.read_csv('../preds/xgb_preds.csv')
lgb_1_4 = pd.read_csv('../preds/lgbm_preds_1.4.csv')
lgb_1_3 = pd.read_csv('../preds/lgbm_preds_1.3.csv')
lgb_1_2 = pd.read_csv('../preds/lgbm_preds_1.2.csv')

# voting = column_shuffle['target']*xgboost['target']*lgb_1_4['target']*lgb_1_3['target']*lgb_1_2['target']
voting = column_shuffle['target']*lgb_1_4['target']*lgb_1_3['target']


print(column_shuffle.head())
print(xgboost.head())
print(lgb_1_4.head())
print(lgb_1_3.head())
print(lgb_1_2.head())

sub_df['target'] = voting

print(sub_df.head())

sub_df.to_csv('../preds/voting_preds_2.csv', index=False)