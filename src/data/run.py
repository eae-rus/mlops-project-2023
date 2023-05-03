import argparse
import pandas as pd
import os
from comtrade import Comtrade
# import typing as tp
from signal_dataset import SignalDataset
from data_preparation import FeatureMaker


def parse_args():
    parser = argparse.ArgumentParser(description='dataset path')

    parser.add_argument('folder_from', metavar='FOLDER_FROM', type=str,
                        help='data folder')
    parser.add_argument('folder_to', metavar='FOLDER_TO', type=str,
                        help='dataset folder')
    parser.add_argument('flag', metavar='FEATURES_MAKER', type=int,
                        help='if need make features')
    return parser.parse_args()


def run():
    args = parse_args()
    files = (f.split('.')[0] for f in os.listdir(args.folder_from) if f.endswith('.cfg'))
    dataset = pd.DataFrame()

    for file in files:
        rec = Comtrade()
        rec.load(os.path.join(args.folder_from, file + '.cfg'),
                 os.path.join(args.folder_from, file + '.dat'),
                 encoding='utf-8')
        for bus in [1, 2]:
            dataset_maker = SignalDataset(file, bus, rec)
            features = dataset_maker.get_features()
            dataset = pd.concat([dataset, features])
    dataset.to_csv(args.folder_to + '/data.csv')

    if args.flag:
        feature_maker = FeatureMaker(dataset)
        fft_df_value = feature_maker.data_preparation_fft()
        fft_df_value.to_csv(args.folder_to + 'data_fft_value.csv')


if __name__ == "__main__":
    run()
