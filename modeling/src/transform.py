import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def feature_engineering(df, features_interaction):
    df['diff'] = df['Time'].diff()

    df_sub = df[features_interaction]
    df = df.drop(features_interaction, axis=1)

    # get feature interactions
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    poly.fit(df_sub)
    columns = poly.get_feature_names_out()
    df_sub = poly.transform(df_sub)
    df_sub = pd.DataFrame(df_sub, columns=columns)

    # drop constant columns
    df_sub = df_sub.loc[:, (df_sub != df_sub.iloc[0]).any()]

    df = pd.concat([df_sub, df], axis=1)

    return df


DATA_PATH = './data/raw/'
PROCESSED_DATA_PATH = './data/processed/'

N_SPLITS = 5
N_TRAIN = 47000
N_VAL = 20000
N_TEST = 29807

DROPLIST = ['Time', 'Class']
FEATURES_FOR_INTERACTION = [
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V7',
    'V9',
    'V10',
    'V11',
    'V12',
    'V13',
    'V16',
    'V17',
    'V18',
]
files = os.listdir(DATA_PATH)


def transform():
    # laod the data
    for index, file in enumerate(files):

        if index == 0:
            df = pd.read_csv(DATA_PATH + file)
        else:
            temp = pd.read_csv(DATA_PATH + file)
            df = df.append(temp, axis=0)

    df = df.sort_values(by='Time')

    # select features
    man_features = pd.DataFrame(FEATURES_FOR_INTERACTION, columns=['0'])
    man_features.to_csv('./data/output_models/features.csv', index=False)

    df = feature_engineering(df=df, features_interaction=FEATURES_FOR_INTERACTION)

    # save files
    # orig_and_gen_fetaures = df.drop(['Time', 'Class'], axis=1).columns.to_list()
    # orig_and_gen_fetaures = pd.DataFrame(orig_and_gen_fetaures, columns=['0'])
    # orig_and_gen_fetaures.to_csv('./experiments/orig_and_gen_features/features.csv', index=False)

    # test subset
    df_test = df.iloc[-N_TEST:, :]
    df_test.to_csv(PROCESSED_DATA_PATH + 'test.csv', index=False)

    df = df.iloc[:-N_TEST, :]

    # cross-validation subsets
    for split in range(N_SPLITS):
        df_train = df.iloc[N_TRAIN * split : N_TRAIN * (split + 1), :]
        df_val = df.iloc[N_TRAIN * (split + 1) : N_TRAIN * (split + 1) + N_VAL, :]

        df_train.to_csv(PROCESSED_DATA_PATH + f'{split}_train.csv', index=False)
        df_val.to_csv(PROCESSED_DATA_PATH + f'{split}_val.csv', index=False)

    return True


if __name__ == '__main__':
    transform()
