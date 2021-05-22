import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./home-credit-default-risk/')
    # Debug mode chỉ đọc 100 dòng đầu cho nhanh
    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--nan_as_cat', action='store_true')
    parser.add_argument('--nrows', type=int, default=None)

    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    return args
