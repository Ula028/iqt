"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from hpsklearn import HyperoptEstimator, random_forest_regression
from hyperopt import hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import utils

print("Loading training data...")
train_lr, train_hr = utils.load_training_data()

# print("Training the imputer...")
# imputer = IterativeImputer(max_iter=10, random_state=0)
# imputer = imputer.fit(train_lr)

print("Training the model...")
# lin_reg = LinearRegression().fit(train_lr, train_hr)
# reg_tree = DecisionTreeRegressor(criterion='mse').fit(train_lr, train_hr)

n_estimators = sample(scope.int(hp.quniform('n_estimators', 5, 15, 1)))
max_depth = sample(scope.int(hp.quniform('max_depth', 10, 50, 1)))
max_features = hp.choice('max_features', ['auto', 'sqrt', 'log2'])
bootstrap = hp.choice('bootstrap', [True, False])

estim = HyperoptEstimator(regressor=random_forest_regression('my_forest', n_estimators=n_estimators,
                                                             max_depth=max_depth, max_features=max_features, bootstrap=bootstrap),
                          max_evals=20, use_partial_fit=True, trial_timeout=10800)
estim.fit(train_lr, train_hr)

print("Loading testing data...")
test_lr, test_hr = utils.load_testing_data()

print(estim.score(test_lr, test_hr))
print(estim.best_model())

# print("Calculating the mean...")
# mean = np.mean(train_lr, axis=0)
# print("Calculating the variance...")
# covariance = np.cov(train_lr, rowvar=False)

# # save the model
# with open('models/lin_reg_model.pickle', 'wb') as handle:
#     pickle.dump(lin_reg.best_model(), handle)

# # save the normal distribution
# with open('models/mean.pickle', 'wb') as handle:
#     pickle.dump(mean, handle)

# with open('models/covariance.pickle', 'wb') as handle:
#     pickle.dump(covariance, handle)

# # save the imputer
# with open('models/imputer.pickle', 'wb') as handle:
#     pickle.dump(imputer, handle)
