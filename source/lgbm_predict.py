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


early_rounds = 3000
num_iterations = 1000000

# Read in data files
train = pd.read_csv('../data/train.csv.zip')
test = pd.read_csv('../data/test.csv.zip')
sub_df = pd.read_csv('../data/sample_submission.csv.zip')

features_to_remove = ['var_15', 'var_174', 'var_150', 'var_139', 'var_127']
train = train.drop(features_to_remove, axis=1)
test = test.drop(features_to_remove, axis=1)

# Drop ID_code
train = train.drop(['ID_code'], axis=1)
test = test.drop(['ID_code'], axis=1)

# # Removing "fake" train rows
# real_vs_fake_train = pd.read_csv('../output/real_vs_fake_train.csv')
# train = train[real_vs_fake_train['target'] < 0.5]

# Split train into features and target
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['ID_code']]
X = X[features]

# Create folds
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Prune to top n most important features
if importance_prune:
    importances = pd.read_csv('../output/lgbm_importance_' + str(version - 0.1) + '.csv')
    features_to_keep = importances['feature'][:num_features_to_keep].to_list()
    X = X[features_to_keep]

# Create arrays for oof predictions and sub predictions
oof_preds = np.zeros(len(X))
sub_preds = np.zeros(test.shape[0])

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    trn_data = lgb.Dataset(X.iloc[trn_][features], label=y.iloc[trn_])
    val_data = lgb.Dataset(X.iloc[val_][features], label=y.iloc[val_])

    clf = lgb.train(params, trn_data, num_iterations, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=early_rounds)

    # Predict on validation fold
    oof_preds[val_] = clf.predict(X.iloc[val_][features], num_iteration=clf.best_iteration)
    if importance_prune:
        sub_preds += clf.predict(test[features_to_keep].values, num_iteration=clf.best_iteration) / num_folds
    else:
        sub_preds += clf.predict(test.values, num_iteration=clf.best_iteration) / num_folds

    print('no {}-fold AUC: {}'.format(fold_ + 1, roc_auc_score(y.iloc[val_].values, oof_preds[val_])))

# Save submission predictions
sub_df['target'] = sub_preds
sub_df.to_csv('../preds/lgbm_preds_' + str(version) + '.csv', index=False)
