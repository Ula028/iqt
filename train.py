"""A script that trains a model for IQT random forest
using previously created dataset.
"""

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import randint
import numpy as np
import pickle


def estimate_random_forest(train_lr, train_hr):
    # n_estimators = sample(scope.int(hp.quniform('n_estimators', 10, 14, 1)))
    # max_depth = sample(scope.int(hp.quniform('max_depth', 40, 50, 1)))
    # min_samples_split = sample(
    #     scope.int(hp.quniform('min_samples_split', 1, 40, 1)))
    # min_samples_leaf = sample(
    #     scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)))
    # max_features = hp.choice('max_features', ['auto', 'sqrt', 'log2'])
    # bootstrap = hp.choice('bootstrap', [True, False])

    evals = 20
    estim = HyperoptEstimator(algo=tpe.suggest, regressor=random_forest_regression('my_forest', bootstrap=False),
                              preprocessing=[], max_evals=evals, trial_timeout=10800)
    estim.fit(train_lr, train_hr)

    return estim.best_model()


def find_reg_tree(train_lr, train_hr):
    reg_tree = DecisionTreeRegressor(
        max_features='sqrt').fit(train_lr, train_hr)

    param_dist = {"max_depth": randint(20, 150),
                  "min_samples_split": randint(5, 50),
                  "min_samples_leaf": randint(1, 25)}

    n_iter_search = 50
    random_search = RandomizedSearchCV(
        reg_tree, param_distributions=param_dist, n_iter=n_iter_search)
    random_search.fit(train_lr, train_hr)

    return random_search.best_estimator_


def train_lin_reg(train_lr, train_hr, datasample_rate):
    print("Training the linear regression model...")
    lin_reg = LinearRegression().fit(train_lr, train_hr)

    # save the model
    with open('models/lin_reg_model' + str(datasample_rate) + '.pickle', 'wb') as handle:
        pickle.dump(lin_reg, handle)

    return lin_reg


def train_reg_tree(train_lr, train_hr, datasample_rate):
    print("Training the decision tree regressor...")
    reg_tree = DecisionTreeRegressor(max_depth=102, max_features='sqrt', min_samples_leaf=24,
                                     min_samples_split=19).fit(train_lr, train_hr)

    # save the model
    with open('models/reg_tree_model' + str(datasample_rate) + '.pickle', 'wb') as handle:
        pickle.dump(reg_tree, handle)

    return reg_tree


def train_ran_forest(train_lr, train_hr, datasample_rate):
    print("Training the random forest...")
    ran_forest = RandomForestRegressor(bootstrap=False, max_features=0.4346383681719076,
                                       n_estimators=50, n_jobs=-1, random_state=1).fit(train_lr, train_hr)

    # save the model
    with open('models/ran_forest_model' + str(datasample_rate) + '.pickle', 'wb') as handle:
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
    rate = 5
    train_lr, train_hr = utils.load_training_data(rate)
    ran_forest = train_ran_forest(train_lr, train_hr, rate)
