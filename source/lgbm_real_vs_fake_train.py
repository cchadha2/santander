import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Data parameters
num_folds = 5
SEED = 5000
version = 1.5
importance_prune = False
importance_save = True
num_features_to_keep = 100
training_dataset = 'train'

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


# early_rounds = 3000
# num_iterations = 1000000
early_rounds = 200
num_iterations = 10000

# Read in data files
train = pd.read_csv('../data/' + training_dataset + '.csv.zip')
test = pd.read_csv('../data/test.csv.zip')
train = train.drop(['ID_code', 'target'], axis=1)
test = test.drop(['ID_code'], axis=1)

set_1 = np.load('../output/public_LB.npy')
set_2 = np.load('../output/private_LB.npy')

real_samples = np.concatenate([set_1, set_2])
real_test = test.iloc[real_samples, :]
real_test['target'] = 1

synthetic_samples = np.load('../output/synthetic_samples_indexes.npy')
fake_test = test.iloc[synthetic_samples, :]
fake_test['target'] = 0

test = pd.concat([real_test, fake_test])

# Split train into features and target
y = test['target']
X = test.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['ID_code']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Create arrays for oof predictions and sub predictions
oof_preds = np.zeros(len(X))
sub_preds = np.zeros(test.shape[0])

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):

    trn_data = lgb.Dataset(X.iloc[trn_], label=y.iloc[trn_])
    val_data = lgb.Dataset(X.iloc[val_], label=y.iloc[val_])


    clf = lgb.train(params, trn_data, num_iterations, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=early_rounds)


    oof_preds[val_] = clf.predict(X.iloc[val_], num_iteration=clf.best_iteration)

    sub_preds += clf.predict(train.values, num_iteration=clf.best_iteration) / num_folds

    score = roc_auc_score(y.iloc[val_].values, oof_preds[val_])
    print('no {}-fold AUC: {}'.format(fold_ + 1, score))

train['target'] = sub_preds
train.to_csv('../output/real_vs_fake_train.csv', index=False)

score = roc_auc_score(y, oof_preds)
print('OVERALL AUC: {:.5f}'.format(score))
