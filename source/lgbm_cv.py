import lightgbm as lgb
import numpy as np
import pandas as pd
import pymongo
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Data parameters
num_folds = 5
# SEED = 5000
# version = 1.0
# importance_prune = False
# num_features_to_keep = 100
training_dataset = 'train'
# # MongoDB parameters
# client = pymongo.MongoClient('mongodb://localhost:27017')
# db = client.reporting
# collection = db.validation
# mongo_save = False
# # Lightgbm parameters
# params = {'num_leaves': 20,
#           'min_data_in_leaf': 60,
#           'objective': 'binary',
#           'max_depth': 8,
#           'learning_rate': 0.02,
#           "boosting": "gbdt",
#           "feature_fraction": 0.8,
#           "bagging_freq": 1,
#           "bagging_fraction": 0.8,
#           "bagging_seed": 11,
#           "metric": 'auc',
#           "lambda_l1": 0.1,
#           "random_state": SEED,
#           "num_iterations": 10000}

# https://www.kaggle.com/fayzur/customer-transaction-prediction-strong-baseline
# Thanks fayzur. Nice Parameter
params = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }

early_rounds = 100

# Create dictionary to send to MongoDB alongside validation score
# mongo_dict = params
# mongo_dict['training_dataset'] = 'train'
# mongo_dict['early_stopping_rounds'] = early_rounds
# mongo_dict['seed'] = SEED
# mongo_dict['num_folds'] = num_folds
# mongo_dict['notes'] = ''

# Read in data files
train = pd.read_csv('../data/' + training_dataset + '.csv.zip')
train = train.drop(['ID_code'], axis=1)

# # Balance dataset
# train_pos = train[train['target']==1]
# train_neg = train[train['target']==0].sample(len(train_pos), random_state=SEED)
# train = pd.concat([train_pos, train_neg], axis=0)

# Split train into features and target
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['ID_code']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2019)

# # Prune to top n most important features
# if importance_prune:
#     importances = pd.read_csv('../output/lgbm_importance_' + str(version - 0.1) + '.csv')
#     features_to_keep = importances['feature'][:num_features_to_keep].to_list()
#     X = X[features_to_keep]

# Create arrays for oof predictions and sub predictions
oof_preds = np.zeros(len(X))
# feature_importance_df = pd.DataFrame()

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    train_x, train_y = X.iloc[trn_], y.iloc[trn_]
    val_x, val_y = X.iloc[val_], y.iloc[val_]

    # sm = SMOTE(random_state=SEED)
    # train_x, train_y = sm.fit_resample(train_x, train_y)

    clf = lgb.LGBMClassifier(**params)
    clf = clf.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=early_rounds, verbose=1000)

    # Predict on validation fold
    oof_preds[val_] = clf.predict(val_x)

    # # Concatenate fold importances into feature importance dataframe
    # fold_importance_df = pd.DataFrame()
    # fold_importance_df["feature"] = X.columns.tolist()
    # fold_importance_df["importance"] = clf.feature_importances_
    # fold_importance_df["fold"] = fold_ + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # print('no {}-fold AUC: {}'.format(fold_ + 1, roc_auc_score(y.iloc[val_].values, oof_preds[val_])))

# score = roc_auc_score(y, oof_preds)
# print('OVERALL AUC: {:.5f}'.format(score))

print("CV score: {:<8.5f}".format(roc_auc_score(y, oof_preds)))

# if mongo_save:
#     mongo_dict['score'] = score
#     collection.insert_one(mongo_dict)

# # Save average feature importances
# feature_importance_df.groupby('feature', as_index=False).mean().drop('fold', axis=1)\
#     .sort_values('importance', ascending=False).to_csv('../output/lgbm_importance_' + str(version) + '.csv',
#                                                        index=False)
