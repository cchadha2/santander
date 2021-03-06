import lightgbm as lgb
import numpy as np
import random
import pandas as pd
import pymongo
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Data parameters
num_folds = 5
SEED = 5000
version = 1.5
importance_prune = False
importance_save = True
num_features_to_keep = 100
training_dataset = 'train'
# MongoDB parameters
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client.reporting
collection = db.validation
mongo_save = True
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


# Create dictionary to send to MongoDB alongside validation score
mongo_dict = params
mongo_dict['training_dataset'] = 'train'
mongo_dict['early_stopping_rounds'] = early_rounds
mongo_dict['num_iterations'] = num_iterations
mongo_dict['seed'] = SEED
mongo_dict['num_folds'] = num_folds
mongo_dict['notes'] = 'removing features with slightly different train and test distributions'

importances = pd.read_csv('../output/lgbm_importance_' + str(version - 0.1) + '.csv')

features_to_remove = ['var_15', 'var_193', 'var_177', 'var_174', 'var_173', 'var_162', 'var_150', 'var_139', 'var_127']

# Read in data files
train = pd.read_csv('../data/' + training_dataset + '.csv.zip')
train = train.drop(['ID_code'], axis=1)
train = train.drop(features_to_remove, axis=1)

# real_vs_fake_train = pd.read_csv('../output/real_vs_fake_train.csv')
# train = train[real_vs_fake_train['target'] < 0.5]

# Split train into features and target
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['ID_code']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Prune to top n most important features
if importance_prune:
    importances = pd.read_csv('../output/lgbm_importance_' + str(version - 0.1) + '.csv')
    features_to_keep = importances['feature'][:num_features_to_keep].to_list()
    X = X[features_to_keep]

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

    score = roc_auc_score(y.iloc[val_].values, oof_preds[val_])
    print('no {}-fold AUC: {}'.format(fold_ + 1, score))

    break
#
# score = roc_auc_score(y, oof_preds)
# print('OVERALL AUC: {:.5f}'.format(score))

importances =  feature_importance_df.groupby('feature', as_index=False).mean().drop('fold', axis=1).sort_values(
                                            'importance', ascending=False)

print(importances.head(10))


if mongo_save:
    mongo_dict['score'] = score
    collection.insert_one(mongo_dict)

if importance_save:
    # Save average feature importances
    importances.to_csv('../output/lgbm_importance_' + str(version) + '.csv', index=False)