import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from data import *
from utils import *
from parameters import get_args


# Light GBM
def kfold_ligthgbm(df, num_folds=5, stratified=True):
    # Trong file data mình đã xử lí test chưa có biến target
    df_train = df[df['TARGET'].notnull()]
    df_test = df[df['TARGET'].isnull()]
    del df
    gc.collect()
    print('Starting LightGBM. Train Shape: {}, Test Shape: {}'.format(df_train.shape, df_test.shape))

    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))   
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(df_train['TARGET'], oof_preds))
    
    df_test['TARGET'] = sub_preds
    df_test[['SK_ID_CURR', 'TARGET']].to_csv("submission_kernel02.csv.csv", index= False)

    return feature_importance_df


