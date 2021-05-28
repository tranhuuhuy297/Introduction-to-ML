import os
import gc
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

from utils import *


def get_raw_df(args, dataset):
    df = pd.read_csv(args.data_path + dataset + '.csv')

    print("Shape of {}: {}".format(dataset, df.shape))
    print("{}: \n{}".format(dataset.dtypes.value_counts()))
    print('\n')
    print(df.select_dtypes('object').apply(pd.Series.nunique))
    explain(df) # Chú thích từng trường dữ liệu
    print('\n')

    return df

def application_train_test(args, nan_as_category=True):
    df_train = pd.read_csv(args.data_path + 'application_train.csv', nrows=args.nrows)
    df_test = pd.read_csv(args.data_path + 'application_test.csv', nrows=args.nrows)

    df_train = df_train.drop(missing_columns(df_train).head(26).index.values, axis=1)

    # Trộn train và test
    df = df_train.copy()
    df = df.append(df_test)

    # Chỉ có 4 bản ghi XNA nên xóa luôn (nhiễu)
    df = df[df['CODE_GENDER'] != 'XNA']

    drop_columns = [f for f in df.columns if 'FLAG_DOC' in f]
    df = df.drop(drop_columns, axis=1)

    # Error value 365243
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    # Create some new features
    df['DIR'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['AIR'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['ACR'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAR'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Encode categorical feature
    df, df_cat = one_hot_encoder(df, nan_as_category)

    # Test has no target
    df_train = df[df.TARGET.notnull()]
    df_test = df[df.TARGET.isnull()]

    poly_fitting_vars = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'DAR']
    df_train[poly_fitting_vars] = df_train[poly_fitting_vars].fillna(df_train[poly_fitting_vars].median())
    df_test[poly_fitting_vars] = df_test[poly_fitting_vars].fillna(df_test[poly_fitting_vars].median())

    poly_feat = PolynomialFeatures(degree=4)

    poly_interaction_train = poly_feat.fit_transform(df_train[poly_fitting_vars])
    poly_interaction_test = poly_feat.transform(df_test[poly_fitting_vars])

    poly_interaction_train = pd.DataFrame(poly_interaction_train, columns=poly_feat.get_feature_names(poly_fitting_vars))
    poly_interaction_test = pd.DataFrame(poly_interaction_test, columns=poly_feat.get_feature_names(poly_fitting_vars))

    poly_interaction_train['TARGET'] = df_train['TARGET']
    poly_interaction_test['TARGET'] = df_test['TARGET']

    interaction = poly_interaction_train.corr()['TARGET'].sort_values()
    selected_inter_variables = list(set(interaction.head(15).index).union(interaction.tail(15).index).difference(set({'1','TARGET'})))
    unselected_cols = [element for element in poly_interaction_train.columns if element not in selected_inter_variables]

    poly_interaction_train = poly_interaction_train.drop(unselected_cols, axis=1)
    poly_interaction_test = poly_interaction_test.drop(unselected_cols, axis=1)

    df_train = df_train.join(poly_interaction_train.drop(['EXT_SOURCE_2'], axis=1))
    df_test = df_test.join(poly_interaction_test.drop(['EXT_SOURCE_2'], axis=1))

    # df_train1 = df_train.fillna(df_train.mean())
    df = df_train.append(df_test)

    df_train1 = df[df.TARGET.notnull()]
    df_test1 = df[df.TARGET.isnull()]

    del df
    gc.collect()

    for i in missing_columns(df_train1).index.values:
        for j in df_train1[df_train1[i].isna()].SK_ID_CURR.values:
            df_train1.drop(df_train1[df_train1['SK_ID_CURR']==j].index, inplace=True)

    for i in df_train1.columns.difference(['TARGET', 'SK_ID_CURR']).values:
        scaler_ = preprocessing.StandardScaler()
        df_train1[i] =  scaler_.fit_transform(pd.DataFrame(df_train1[i]))
        df_test1[i] = scaler_.transform(pd.DataFrame(df_test1[i]))

    return df_train1, df_test1

def bureau(args, nan_as_category=True):
    df_bureau = pd.read_csv(args.data_path + 'bureau.csv', nrows=args.nrows)
    df_bureau_balance = pd.read_csv(args.data_path + 'bureau_balance.csv', nrows=args.nrows)

    # Encode categorical feature
    bureau, bureau_cat = one_hot_encoder(df_bureau, nan_as_category)
    bb, bb_cat = one_hot_encoder(df_bureau_balance, nan_as_category)

    # Tính mean của các lần bureau trướ
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    # Drop cột id bureau để tí merge với application
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})  
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Set 1 cho active trong bureau
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    del active, active_agg
    gc.collect()

    # Set 0 cho closed trong bureau, 2 giá trị còn lại không quan trọng lắm
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg
    gc.collect()

    bureau_agg.drop(missing_columns(bureau_agg).head(32).index.values, axis=1, inplace=True)
    bureau_agg.fillna(bureau_agg.mean())

    return bureau_agg
    

def previous_application(args, nan_as_category=True):
    df_previous_application = pd.read_csv(args.data_path + 'previous_application.csv', nrows=args.nrows)

    prev, prev_cat = one_hot_encoder(df_previous_application, nan_as_category)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in prev_cat:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    
    return prev_agg


def pos_cash(args, nan_as_category=True):
    pos = pd.read_csv(args.data_path + 'POS_CASH_balance.csv', nrows=args.nrows)
    pos, pos_cat = one_hot_encoder(pos, nan_as_category)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in pos_cat:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    del pos
    gc.collect()

    return pos_agg


def installments_payments(args, nan_as_category=True):
    ins = pd.read_csv(args.data_path + 'installments_payments.csv', nrows=args.nrows)
    ins, ins_cat = one_hot_encoder(ins, nan_as_category=True)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }

    for cat in ins_cat:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    del ins
    gc.collect()

    return ins_agg


def credit_card_balance(args, nan_as_category=True):
    cc = pd.read_csv(args.data_path + 'credit_card_balance.csv', nrows=args.nrows)
    cc, cc_cat   = one_hot_encoder(cc, nan_as_category)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc
    gc.collect()

    return cc_agg


def data_explain(args):
    # Application Main File
    df_train = get_raw_df(args, 'application_train')
    print(missing_columns(df_train)['Missing Count %'])
    print('-------------------------------------------------------------------------')
    df_test = get_raw_df(args, 'application_test')
    print(missing_columns(df_test)['Missing Count %'])
    print('-------------------------------------------------------------------------')

    # Bureau File
    df_bureau = get_raw_df(args, 'bureau')
    print(missing_columns(df_bureau)['Missing Count %'])
    print('-------------------------------------------------------------------------')
    df_bureau_balance = get_raw_df(args, 'bureau_balance')
    print(missing_columns(df_bureau_balance)['Missing Count %'])
    print('-------------------------------------------------------------------------')

    # Previous Application
    df_previous_application = get_raw_df(args, 'previous_application')
    print(missing_columns(df_previous_application)['Missing Count %'])
    print('-------------------------------------------------------------------------')

    df_pos_cash_balance = get_raw_df(args, 'POS_CASH_balance')
    print(missing_columns(df_pos_cash_balance)['Missing Count %'])
    print('-------------------------------------------------------------------------')
    df_installments_payments = get_raw_df(args, 'installments_payments')
    print(missing_columns(df_installments_payments)['Missing Count %'])
    print('-------------------------------------------------------------------------')
    df_credit_card_balance = get_raw_df(args, 'credit_card_balance')
    print(missing_columns(df_credit_card_balance)['Missing Count %'])
    print('-------------------------------------------------------------------------')


def data_debug(args):
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
    df_train, df_test = application_train_test(args, nan_as_category=args.nan_as_cat)
    df = df_train.append(df_test)

    # Bureau
    df_bureau = bureau(args, nan_as_category=args.nan_as_cat)
    df = df.join(df_bureau, how='left', on='SK_ID_CURR')
    del df_bureau
    gc.collect()

    # Previous Apllication
    df_prev = previous_application(args, nan_as_category=args.nan_as_cat)
    df = df.join(df_prev, how='left', on='SK_ID_CURR')
    del df_prev
    gc.collect()

    # POS cash
    df_pos_cash = pos_cash(args, nan_as_category=args.nan_as_cat)
    df = df.join(df_pos_cash, how='left', on='SK_ID_CURR')
    del df_pos_cash
    gc.collect()

    # Install payment
    df_ins_pay = installments_payments(args, nan_as_category=args.nan_as_cat)
    df = df.join(df_ins_pay, how='left', on='SK_ID_CURR')
    del df_ins_pay
    gc.collect()

    # Credit Card
    df_credit = credit_card_balance(args, nan_as_category=args.nan_as_cat)
    df = df.join(df_credit, how='left', on='SK_ID_CURR')
    del df_credit
    gc.collect()

    return df