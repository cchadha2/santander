import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../data/train.csv.zip')
train = train.drop(['ID_code', 'target'], axis=1)


lgb_shuffle = pd.read_csv('../output/lgbm_oof_preds_shuffling_4.csv', usecols=['preds'])
# xgboost = pd.read_csv('../output/xgb_oof_preds.csv', usecols=['preds'])
# cat = pd.read_csv('../output/cat_oof_preds.csv', usecols=['preds'])
# bayes = pd.read_csv('../output/lgbm_oof_preds.csv', usecols=['preds'])
svm = pd.read_csv('../output/linear_svm_oof_preds.csv', usecols=['prediction'])
# lgb_1 = pd.read_csv('../output/lgbm_oof_preds_1.csv', usecols=['preds'])
lgb_original = pd.read_csv('../output/lgbm_oof_preds.csv', usecols=['target', 'preds'])

oof_df = lgb_original.drop('preds', axis=1)

# oof_df = lgb_original.rename(columns={'preds': 'preds_1'})
oof_df['preds_2'] = lgb_shuffle['preds']
# oof_df['preds_2'] = xgboost['preds']
# oof_df['preds_3'] = cat['preds']
oof_df['preds_4'] = svm['prediction']
# oof_df['preds_5'] = bayes['preds']
# oof_df['preds_7'] = lgb_1['preds']


print(oof_df.head())

# del lgb_shuffle, xgboost, lgb_original, cat, svm, bayes, lgb_1

# print(roc_auc_score(oof_df['target'],
#                     oof_df['preds_1']*oof_df['preds_2']*oof_df['preds_3']\
#                     *oof_df['preds_4']*oof_df['preds_5']*oof_df['preds_6'] \
#                     *oof_df['preds_7']))

# print(roc_auc_score(oof_df['target'], oof_df['preds_1']*oof_df['preds_2']*oof_df['preds_3']*oof_df['preds_4']*oof_df['preds_5']))
#
# print(roc_auc_score(oof_df['target'],
#         0.4*oof_df['preds_1'] + 0.15*oof_df['preds_2'] + 0.15*oof_df['preds_3'] + 0.15*oof_df['preds_4'] + 0.15*oof_df['preds_5']))
#
# print(roc_auc_score(oof_df['target'],
#         0.5*oof_df['preds_1'] + 0.1*oof_df['preds_2'] + 0.1*oof_df['preds_3'] + 0.1*oof_df['preds_4'] + 0.1*oof_df['preds_5']))
#


# print(roc_auc_score(oof_df['target'], oof_df['preds_1']*oof_df['preds_2']*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.33*oof_df['preds_1'] + 0.33*oof_df['preds_2']+ 0.33*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.4*oof_df['preds_1'] + 0.3*oof_df['preds_2']+ 0.3*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.5*oof_df['preds_1'] + 0.25*oof_df['preds_2']+ 0.25*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.6*oof_df['preds_1'] + 0.2*oof_df['preds_2']+ 0.2*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.7*oof_df['preds_1'] + 0.15*oof_df['preds_2']+ 0.15*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.8*oof_df['preds_1'] + 0.1*oof_df['preds_2']+ 0.1*oof_df['preds_3']))
#
# print(roc_auc_score(oof_df['target'], 0.6*oof_df['preds_1'] + 0.4*oof_df['preds_2']))
#
# print(roc_auc_score(oof_df['target'], 0.7*oof_df['preds_1'] + 0.3*oof_df['preds_2']))
#
# print(roc_auc_score(oof_df['target'], 0.8*oof_df['preds_1'] + 0.2*oof_df['preds_2']))
#
# print(roc_auc_score(oof_df['target'], 0.9*oof_df['preds_1'] + 0.1*oof_df['preds_2']))

# Create arrays for oof predictions and sub predictions
# oof_preds = np.zeros(len(oof_df))
oof_preds_log = np.zeros(len(oof_df))

X = oof_df.drop(columns=['target'], axis=1)
y = oof_df['target']


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=5000)

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(X.iloc[trn_], y.iloc[trn_])
    oof_preds_log[val_] = clf.predict_proba(X.iloc[val_])[:,1]
    oof_preds_log[val_] = (oof_preds_log[val_] - oof_preds_log[val_].min()) / (oof_preds_log[val_].max() - oof_preds_log[val_].min())


    print('no {}-fold AUC: {}'.format(fold_ + 1, roc_auc_score(y.iloc[val_].values, oof_preds_log[val_])))
#
print('OVERALL AUC: {:.5f}'.format(roc_auc_score(y, oof_preds_log)))