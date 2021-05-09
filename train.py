import os
import IPython
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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