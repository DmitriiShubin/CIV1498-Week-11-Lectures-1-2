# import
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from bayes_opt import BayesianOptimization
from utils import target_metric
from utils.data_generator import DataGenerator


class LightGBM:
    def __init__(self, experiment_hparams, optmizer_hparams, model_hparams=None):

        # load features that will be used for this model
        features = pd.read_csv(experiment_hparams['feature_csv'])['0'].to_list()
        self.data_generator = DataGenerator(features=features)

        self.optmizer_hparams = optmizer_hparams
        for i in self.optmizer_hparams.keys():
            try:
                self.optmizer_hparams[i] = eval(self.optmizer_hparams[i])
            except:
                pass

        self.experiment_hparams = experiment_hparams

        if model_hparams is not None:
            self.model_hparams = model_hparams

    # hyperparameter optimization. predictive model
    def hyperopt(self):
        def model_eval_train(
            num_leaves,
            min_data_in_leaf,
            max_depth,
            feature_fraction,
            bagging_freq,
            bagging_fraction,
            lambda_l1,
            lambda_l2,
            min_split_gain,
            min_child_weight,
        ):
            self.model_hparams = {
                'num_leaves': int(round(num_leaves)),
                'min_data_in_leaf': int(round(min_data_in_leaf)),
                'feature_fraction': max(min(feature_fraction, 1), 0),
                'max_depth': int(round(max_depth)),
                "bagging_freq": int(round(bagging_freq)),
                "bagging_fraction": round(min((max(bagging_fraction, 0), 1)), 1),
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "min_split_gain": min_split_gain,
                "min_child_weight": min_child_weight,
                'objective': self.optmizer_hparams['objective'],
                "boosting": self.optmizer_hparams['boosting'],
                "verbosity": -1,
                "metric": self.optmizer_hparams['metric'],
                'learning_rate': self.optmizer_hparams['lr'],
                'seed': 42,
                'bagging_seed': 42,
                'drop_seed': 42,
            }

            return self.run_cv()[0]

        # apply hyperopt
        lgbBO = BayesianOptimization(
            model_eval_train,
            {
                'num_leaves': self.optmizer_hparams['num_leaves'],
                'min_data_in_leaf': self.optmizer_hparams['min_data_in_leaf'],
                'max_depth': self.optmizer_hparams['max_depth'],
                'feature_fraction': self.optmizer_hparams['feature_fraction'],
                'bagging_fraction': self.optmizer_hparams['bagging_fraction'],
                'bagging_freq': self.optmizer_hparams['bagging_freq'],
                'lambda_l1': self.optmizer_hparams['lambda_l1'],
                'lambda_l2': self.optmizer_hparams['lambda_l2'],
                'min_split_gain': self.optmizer_hparams['min_split_gain'],
                'min_child_weight': self.optmizer_hparams['min_child_weight'],
            },
            random_state=0,
        )
        lgbBO.maximize(
            init_points=self.optmizer_hparams['init_round'], n_iter=self.optmizer_hparams['opt_round']
        )

        # get the list of parameters
        hyperopt_params = lgbBO.max['params']

        self.model_hparams = {
            'num_leaves': int(round(hyperopt_params['num_leaves'])),
            'min_data_in_leaf': int(round(hyperopt_params['min_data_in_leaf'])),
            'feature_fraction': max(min(hyperopt_params['feature_fraction'], 1), 0.001),
            'max_depth': int(round(hyperopt_params['max_depth'])),
            "bagging_freq": int(round(hyperopt_params['bagging_freq'])),
            "bagging_fraction": round(min((max(hyperopt_params['bagging_fraction'], 0.001), 1)), 1),
            "lambda_l1": hyperopt_params['lambda_l1'],
            "lambda_l2": hyperopt_params['lambda_l2'],
            "min_split_gain": hyperopt_params['min_split_gain'],
            "min_child_weight": hyperopt_params['min_child_weight'],
            'objective': self.optmizer_hparams['objective'],
            "boosting": "gbdt",
            "verbosity": -1,
            "metric": self.optmizer_hparams['metric'],
            'learning_rate': self.optmizer_hparams['lr'],
            'seed': 42,
            'bagging_seed': 42,
            'drop_seed': 42,
        }

        return True

    # model cross-validation
    def run_cv(self, save_model=False):

        val_scores = []
        val_thresholds = []
        test_scores = []
        test_thresholds = []

        for fold in range(self.data_generator.n_splits):

            train, target_train, val, target_val = self.data_generator.get_train_val(fold=fold)
            test, target_test = self.data_generator.get_test()

            # get predictions
            self.train_fold(train=train, val=val, target_train=target_train, target_val=target_val)

            pred_val = self.predict(test=val)
            pred_test = self.predict(test=test)

            # calculate the target metric
            val_score, val_threshold = target_metric(y_pred=pred_val, y_gt=target_val)
            val_scores.append(val_score)
            val_thresholds.append(val_threshold)

            test_score, test_threshold = target_metric(y_pred=pred_test, y_gt=target_test)
            test_scores.append(test_score)
            test_thresholds.append(test_threshold)

            print('Current fold is: ', fold)
            print('Current val metric: ', val_scores[-1])
            print('Current test metric: ', test_scores[-1])

            if save_model:
                pickle.dump(
                    self.model, open(self.experiment_hparams['result_path'] + f'{fold}_lgb_model.pkl', 'wb')
                )

        return (
            np.mean(val_scores),
            val_scores,
            val_thresholds,
            test_scores,
            test_thresholds,
        )

    def train_fold(self, train, val, target_train, target_val):

        # fix random seed
        np.random.seed(42)

        self.model = self.crate_model(param=self.model_hparams, num_round=self.optmizer_hparams['num_round'])

        # train the model
        self.model.fit(
            train,
            target_train,
            eval_metric=self.model_hparams['metric'],
            eval_set=[(val, target_val)],
            verbose=self.optmizer_hparams['verbosity'],
            early_stopping_rounds=self.optmizer_hparams['early_stopping'],
        )

        return True

    def predict(self, test):
        # make predictions
        predictions = self.model.predict(test, raw_score=True)
        return self.sigmoid(predictions)

    ######### utils ###########

    # activation function

    def crate_model(self, param, num_round):

        # set parameters of the lightgbm model
        model = lgb.LGBMClassifier(
            task='train',
            num_leaves=param['num_leaves'],
            min_data_in_leaf=param['min_data_in_leaf'],
            feature_fraction=param['feature_fraction'],
            max_depth=param['max_depth'],
            bagging_freq=param['bagging_freq'],
            bagging_fraction=param['bagging_fraction'],
            lambda_l1=param['lambda_l1'],
            lambda_l2=param['lambda_l2'],
            min_split_gain=param['min_split_gain'],
            min_child_weight=param['min_child_weight'],
            objective=param['objective'],
            boosting=param['boosting'],
            verbose=param['verbosity'],
            metric=param['metric'],
            learning_rate=param['learning_rate'],
            seed=param['seed'],
            bagging_seed=param['bagging_seed'],
            drop_seed=param['drop_seed'],
            n_estimators=num_round,
            num_threads=psutil.cpu_count(logical=False),
        )

        return model

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))
