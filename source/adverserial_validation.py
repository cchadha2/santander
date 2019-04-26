import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Data parameters
num_folds = 5
SEED = 5000

# Lightgbm parameters
params = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}


early_rounds = 200
num_iterations = 10000


# Read in data files
train = pd.read_csv('../data/train.csv.zip')
test = pd.read_csv('../data/test.csv.zip')
train = train.drop(['ID_code'], axis=1)
test = test.drop(['ID_code'], axis=1)

train['target'] = 0
test['target'] = 1

set_1 = np.load('../output/public_LB.npy')
set_2 = np.load('../output/private_LB.npy')
synthetic_samples = np.load('../output/synthetic_samples_indexes.npy')

test = test.drop(synthetic_samples)
test = test.drop(set_1)

train = pd.concat([train, test])

# Split train into features and target
y = train['target']
X = train.drop('target', axis=1)
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Create arrays for oof predictions and sub predictions
oof_preds = np.zeros(len(X))
feature_importance_df = pd.DataFrame()

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):

    trn_data = lgb.Dataset(X.iloc[trn_], label=y.iloc[trn_])
    val_data = lgb.Dataset(X.iloc[val_], label=y.iloc[val_])


    clf = lgb.train(params, trn_data, num_iterations, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=early_rounds)


    oof_preds[val_] = clf.predict(X.iloc[val_], num_iteration=clf.best_iteration)


    # Concatenate fold importances into feature importance dataframe
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X.columns.tolist()
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('no {}-fold AUC: {}'.format(fold_ + 1, roc_auc_score(y.iloc[val_].values, oof_preds[val_])))

print('OVERALL AUC: {:.5f}'.format(roc_auc_score(y, oof_preds)))

train['preds'] = oof_preds
train.to_csv('../output/adverserial_validation.csv',
             index=False)

# Save average feature importances
feature_importance_df.groupby('feature', as_index=False).mean().drop('fold', axis=1).sort_values(
                     'importance', ascending=False).to_csv('../output/av_lgbm_importance.csv',
                     index=False)
