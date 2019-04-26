import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sub_df = pd.read_csv('../data/sample_submission.csv')

lgb_shuffle_train = pd.read_csv('../output/lgbm_oof_preds_shuffling_4.csv', usecols=['target', 'preds'])
svm_train = pd.read_csv('../output/linear_svm_oof_preds.csv', usecols=['prediction'])

oof_df = lgb_shuffle_train
oof_df['preds_2'] = svm_train

X = oof_df.drop(columns=['target'], axis=1)
y = oof_df['target']

lgb_shuffle = pd.read_csv('../preds/lgbm_preds_shuffling_4.csv', usecols=['target'])
svm = pd.read_csv('../preds/submission_5x-LinearSVC-01-v1_085956_2019-03-22-22-40.csv', usecols=['target'])

sub_df['preds'] = lgb_shuffle
sub_df['preds_2'] = svm

test = sub_df.drop(['ID_code', 'target'], axis=1)

print(sub_df.head())

subs = np.zeros(len(test))


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=5000)

for fold_, (trn_, val_) in enumerate(folds.split(X, y)):

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(X.iloc[trn_], y.iloc[trn_])

    subs+= clf.predict_proba(test.values)[:, 1]


sub_df['target'] = subs
sub_df = sub_df.drop(['preds', 'preds_2'], axis=1)
sub_df.to_csv('../preds/logistic_regression_ensembling_1.csv', index=False)