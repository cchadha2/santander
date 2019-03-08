#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-organizations/141/thumbnail.jpg?r=890)
# # Santander Customer Transaction Prediction
# Can you identify who will make a transaction?
# 
# Version6
# - Ensemble : LB 0.899
# - LightGBM : LB 0.898
# - Catboost : LB 0.898 

# In[18]:


### 패키지 설치 
import pandas as pd #Analysis
import warnings 
warnings.filterwarnings('ignore')
import random
import numpy as np 
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


train_df = pd.read_csv("../data/train.csv.zip")
test_df = pd.read_csv("../data/test.csv.zip")

# ## LightGBM BaseLine

# In[29]:


# https://www.kaggle.com/fayzur/customer-transaction-prediction-strong-baseline
# Thanks fayzur. Nice Parameter 
param = {
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


# In[30]:


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


# In[ ]:
from sklearn.metrics import roc_auc_score, roc_curve

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / 5

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


