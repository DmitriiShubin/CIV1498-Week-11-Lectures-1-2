# import libs
import os

import numpy as np
import pandas as pd
from config_pipeline import config


# DATA GENERATOR
class DataGenerator:
    def __init__(self, features):

        # fix random seed
        np.random.seed(42)

        self.config = config['data_generator_params']

        self.train_name = self.config['train_name']
        self.test_name = self.config['test_name']
        self.val_name = self.config['val_name']

        self.data_path = self.config['data_path']
        self.target = self.config['target']
        self.features = features

        self.n_splits = len([i for i in os.listdir(self.data_path) if i.find('train') != -1])

        self.load_data()

    def load_data(self):

        self.train_splits = {}
        for split in range(self.n_splits):
            self.train_splits[split] = self.preprocessing_train(
                pd.read_csv(
                    self.data_path + str(split) + "_" + self.train_name,
                    index_col=None,
                )
            )

        # val set
        self.val_splits = {}
        for split in range(self.n_splits):
            self.val_splits[split] = self.preprocessing(
                pd.read_csv(
                    self.data_path + str(split) + "_" + self.val_name,
                    index_col=None,
                )
            )

        # test set
        self.df_test = pd.read_csv(
            self.data_path + self.test_name,
            index_col=None,
        )
        self.df_test = self.preprocessing(self.df_test)

        return True

    def preprocessing_train(self, df):

        target = df[self.target]
        df = df[self.features]

        # upsampling
        columns = df.columns
        df, target = self.upsampling(df.values, target.values)
        df = pd.DataFrame(df, columns=columns)

        df[self.target] = target

        return df

    def preprocessing(self, df):

        df = df[self.features + [self.target]]

        return df

    def get_train_val(self, fold):
        return (
            self.train_splits[fold].drop(self.target, axis=1).values,
            self.train_splits[fold][self.target].values,
            self.val_splits[fold].drop(self.target, axis=1).values,
            self.val_splits[fold][self.target].values,
        )

    def get_test(self):
        return self.df_test.drop(self.target, axis=1).values, self.df_test[self.target].values

    # upsampling to make class balancing equal
    def upsampling(self, X_loc, y_loc):

        UniqClass = np.unique(y_loc)

        mostQreq = 0
        numSam_max = 0

        for i in UniqClass:
            numSam = np.where(y_loc == i)[0].shape[0]
            if numSam_max < numSam:
                numSam_max = numSam
                mostQreq = i

        for i in UniqClass:
            if i == mostQreq:
                continue
            else:
                # applying of upsampling trainng set
                X_US = np.zeros((numSam_max - np.where(y_loc == i)[0].shape[0], X_loc.shape[1]))
                X_minor = X_loc[np.where(y_loc == i)[0]]
                y_minor = np.zeros((X_US.shape[0]))
                y_minor[:] = i

                for j in range(X_US.shape[0]):
                    ind = np.random.randint(0, X_minor.shape[0])
                    X_US[j, :] = X_minor[ind, :]

                X_loc = np.concatenate((X_loc, X_US))
                y_loc = np.concatenate((y_loc, y_minor))

        # random permutation
        temp = np.zeros((X_loc.shape[0], X_loc.shape[1] + 1))
        temp[:, 0 : X_loc.shape[1]] = X_loc
        temp[:, X_loc.shape[1]] = y_loc[:]

        temp = np.take(temp, np.random.permutation(temp.shape[0]), axis=0, out=temp)

        X_loc = temp[:, 0 : X_loc.shape[1]]
        y_loc[:] = temp[:, X_loc.shape[1]]

        return X_loc, y_loc
