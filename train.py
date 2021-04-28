"""A script that trains a model for IQT random forest
using previously created dataset.
"""
from hpsklearn.components import _trees_min_samples_leaf
import utils
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope
from hyperopt import hp
from hpsklearn import HyperoptEstimator, random_forest_regression, decision_tree
import numpy as np
import pickle
import os
os.environ['OMP_NUM_THREADS'] = '1'


def estimate_random_forest(train_lr, train_hr):
    n_estimators = sample(scope.int(hp.quniform('n_estimators', 10, 20, 1)))
    max_depth = sample(scope.int(hp.quniform('max_depth', 30, 70, 1)))
    max_features = hp.choice('max_features', ['auto', 'sqrt', 'log2'])
    bootstrap = hp.choice('bootstrap', [True, False])

    estim = HyperoptEstimator(regressor=random_forest_regression(
        'my_forest', n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt'), max_evals=20, trial_timeout=10800)
    estim.fit(train_lr, train_hr)

    return estim.best_model()


def estimate_reg_tree(train_lr, train_hr):
    max_depth = sample(scope.int(hp.quniform('max_depth', 30, 70, 1)))
    max_features = hp.choice('max_features', ['auto', 'sqrt', 'log2'])
    min_samples_split = sample(
        scope.int(hp.quniform('min_samples_split', 1, 40, 1)))
    min_samples_leaf = sample(
        scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)))

    estim = HyperoptEstimator(regressor=decision_tree(
        'my_tree', max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf), max_evals=25, trial_timeout=10800)
    estim.fit(train_lr, train_hr)

    return estim.best_model()


def train_iterative_imputer(train_lr):
    print("Training the imputer...")
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputer = imputer.fit(train_lr)

    # save the imputer
    with open('models/imputer.pickle', 'wb') as handle:
        pickle.dump(imputer, handle)

    return imputer


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
    ran_forest = RandomForestRegressor(
        max_depth=45, max_features='sqrt', n_estimators=14, n_jobs=-1, random_state=1).fit(train_lr, train_hr)

    # save the model
    with open('models/ran_forest_model.pickle', 'wb') as handle:
        pickle.dump(ran_forest, handle)

    return ran_forest


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
    best_model = estimate_reg_tree(train_lr, train_hr)
    print(best_model)
