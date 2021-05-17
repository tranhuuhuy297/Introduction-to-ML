import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data import *
from utils import *
from parameters import get_args


def main(args, **model_kwargs):
    if (args.explain): data_explain(args) # Giải thích các trường thông tin cho các file cho anh em xem (có translate)
    if (args.debug): data_debug(args) # Chạy debug xem xử lý data oke không
    df = read_data(args)
    df.to_csv('data.csv', index=False)

if __name__ == "__main__":
    args = get_args()
    main(args)