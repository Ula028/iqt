"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from hyperopt import hp, tpe
from hpsklearn import HyperoptEstimator, random_forest_regression, min_max_scaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def estimate_random_forest(train_lr, train_hr):
    n_estimators = sample(scope.int(hp.quniform('n_estimators', 10, 14, 1)))
    max_depth = sample(scope.int(hp.quniform('max_depth', 40, 50, 1)))
    min_samples_split = sample(
        scope.int(hp.quniform('min_samples_split', 1, 40, 1)))
    min_samples_leaf = sample(
        scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)))
    max_features = hp.choice('max_features', ['auto', 'sqrt', 'log2'])
    bootstrap = hp.choice('bootstrap', [True, False])

    estim = HyperoptEstimator(algo=tpe.suggest, regressor=random_forest_regression(
        'my_forest', n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt', min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=False), preprocessing=[min_max_scaler('my_scaler')], max_evals=25, trial_timeout=10800)
    estim.fit(train_lr, train_hr)

    return estim.best_model()


def train_lin_reg(train_lr, train_hr):
    print("Training the linear regression model...")
    lin_reg = LinearRegression().fit(train_lr, train_hr)

    # save the model
    with open('models/lin_reg_model.pickle', 'wb') as handle:
        pickle.dump(lin_reg, handle)

    return lin_reg


def train_reg_tree(train_lr, train_hr):
    print("Training the decision tree regressor...")
    reg_tree = DecisionTreeRegressor(criterion='mse').fit(train_lr, train_hr)

    # save the model
    with open('models/reg_tree_model.pickle', 'wb') as handle:
        pickle.dump(reg_tree, handle)

    return reg_tree


def train_ran_forest(train_lr, train_hr):
    print("Training the decision tree regressor...")
    # ran_forest = RandomForestRegressor(n_estimators=10, max_depth=43, min_samples_split=11, min_samples_leaf=19, max_features='sqrt', bootstrap=False).fit(train_lr, train_hr)

    ran_forest = RandomForestRegressor(n_estimators=100, max_depth=50, max_features='sqrt', bootstrap=True, n_jobs=-1).fit(train_lr, train_hr)

    # save the model
    with open('models/ran_forest_model.pickle', 'wb') as handle:
        pickle.dump(ran_forest, handle)

    return ran_forest

def fit_scaler(train_lr):
    scaler = MinMaxScaler()
    scaler.fit(train_lr)
    
    # save the scaler
    with open('models/min_max_scaler.pickle', 'wb') as handle:
        pickle.dump(scaler, handle)
        
    return scaler


def calculate_gaussian(train_lr):
    print("Calculating the mean...")
    mean = np.mean(train_lr, axis=0)
    print("Calculating the variance...")
    covariance = np.cov(train_lr, rowvar=False)

    # save the normal distribution
    with open('models/mean.pickle', 'wb') as handle:
        pickle.dump(mean, handle)

    with open('models/covariance.pickle', 'wb') as handle:
        pickle.dump(covariance, handle)

    return mean, covariance


if __name__ == "__main__":
    train_lr, train_hr = utils.load_training_data()
    # scaler = utils.load_scaler()  
    # scaler.transform(train_lr)
    ran_forest = train_ran_forest(train_lr, train_hr)