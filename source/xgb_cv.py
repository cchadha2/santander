import xgboost as xgb
import numpy as np
import pandas as pd
import pymongo
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
# MongoDB parameters
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client.reporting
collection = db.xgb_validation
mongo_save = True

# XGBoost parameters
params = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': 8,
    'eval_metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary:logistic',
    'verbosity': 1
}


early_rounds = 10
num_iterations = 1

# Create dictionary to send to MongoDB alongside validation score
mongo_dict = params
# mongo_dict = {}
mongo_dict['training_dataset'] = 'train'
mongo_dict['early_stopping_rounds'] = early_rounds
mongo_dict['num_iterations'] = num_iterations
mongo_dict['seed'] = SEED
mongo_dict['num_folds'] = num_folds
mongo_dict['notes'] = 'new kaggle params'

# Read in data files
train = pd.read_csv('../data/' + training_dataset + '.csv.zip')
train = train.drop(['ID_code'], axis=1)

# Split train into features and target
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['ID_code']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Prune to top n most important features
if importance_prune:
    importances = pd.read_csv('../output/xgb_importance_' + str(version - 0.1) + '.csv')
    features_to_keep = importances['feature'][:num_features_to_keep].to_list()
    X = X[features_to_keep]

# Create arrays for oof predictions and sub predictions
oof_preds = np.zeros(len(X))
feature_importance_df = pd.DataFrame()

# Split data into folds and train
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    trn_data = xgb.DMatrix(X.iloc[trn_], label=y.iloc[trn_])
    val_data = xgb.DMatrix(X.iloc[val_], label=y.iloc[val_])

    clf = xgb.train(params,
                    trn_data,
                    num_iterations,
                    evals=[(trn_data, 'train'), (val_data, 'validation')],
                    early_stopping_rounds=early_rounds)

    oof_preds[val_] = clf.predict(val_data,  ntree_limit=clf.best_ntree_limit)

    importances = clf.get_score()

    # Concatenate fold importances into feature importance dataframe
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = importances.keys()
    fold_importance_df["importance"] = importances.values()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    del clf

    print('no {}-fold AUC: {}'.format(fold_ + 1, roc_auc_score(y.iloc[val_].values, oof_preds[val_])))

score = roc_auc_score(y, oof_preds)
print('OVERALL AUC: {:.5f}'.format(score))

feature_importance_df.groupby('feature', as_index=False).mean().drop('fold', axis=1).sort_values(
    'importance', ascending=False).to_csv('../output/xgb_importance_' + '.csv',
                                          index=False)