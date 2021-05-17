import numpy as np
import pandas as pd


def explain(df):
    df_explain = pd.read_csv('./ML.csv')
    for i in df.columns.values: 
        if (i not in df_explain.Row.values): continue
        print(i, ' --- ', df_explain[df_explain.Row==i]['Tiếng mẹ đẻ'].values[0])
        if (len(df[i].value_counts()) < 10):
            for k, j in df[i].value_counts().items():
                print('   ', k, '   ', j)
        print('-----------------------------')


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype=='object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


