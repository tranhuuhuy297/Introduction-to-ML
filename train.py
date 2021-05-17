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

args = get_args()

if args.explain:
    # Application Main File
    df_train = get_raw_df(args, 'application_train')
    df_test = get_raw_df(args, 'application_test')

    # Bureau File
    df_bureau = get_raw_df(args, 'bureau')
    df_bureau_balance = get_raw_df(args, 'bureau_balance')

    # Previous Application
    df_previous_application = get_raw_df(args, 'previous_application')

    df_pos_cash_balance = get_raw_df(args, 'POS_CASH_balance')
    df_installments_payments = get_raw_df(args, 'installments_payments')
    df_credit_card_balance = get_raw_df(args, 'credit_card_balance')

if args.debug:
    print('-----Debug mod, only read 100 rows')
    args.nrows = 100
    if not os.path.exists(os.path.join(os.getcwd(), 'debug_file')): os.mkdir('debug_file')

    application_train_test(args).to_csv('debug_file/' + 'application.csv', index=False)
    print('     Done Application')
    bureau(args).to_csv('debug_file/' + 'bureau.csv', index=False)
    print('     Done Bureau')
    previous_application(args).to_csv('debug_file/' + 'prev.csv')
    print('     Done Previous Application')
    pos_cash(args).to_csv('debug_file/' + 'pos_cash.csv')
    print('     Done POS Cash')
    installments_payments(args).to_csv('debug_file/' + 'install_payment.csv')
    print('     Done Install Payment')
    credit_card_balance(args).to_csv('debug_file/' + 'credit_card.csv')
    print('     Done Credit Card')

def read_data(args):
    df = application_train_test(args, nan_as_category=True)

    # Bureau
    df_bureau = bureau(args)
    df = df.join(df_bureau, how='left', on='SK_ID_CURR')
    del df_bureau
    gc.collect()

    # Previous Apllication
    df_prev = previous_application(args)
    df = df.join(df_prev, how='left', on='SK_ID_CURR')
    del df_prev
    gc.collect()

    # POS cash
    df_pos_cash = pos_cash(args)
    df = df.join(df_pos_cash, how='left', on='SK_ID_CURR')
    del df_pos_cash
    gc.collect()

    # Install payment
    df_ins_pay = installments_payments(args)
    df = df.join(df_ins_pay, how='left', on='SK_ID_CURR')
    del df_ins_pay
    gc.collect()

    # Credit Card
    df_credit = credit_card_balance(args)
    df = df.join(df_credit, how='left', on='SK_ID_CURR')
    del df_credit
    gc.collect()

    return df

def kfold_ligthgbm(df, num_folds=5, stratied=True):
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

    return feature_importance_df

df = read_data(args)
df_feature_inportance = kfold_ligthgbm(df)