"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

def load_training_data():
    dict_data = np.load('preprocessed_data/patches.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    return patches_lr, patches_hr

def train_linear_regression(patches_lr, patches_hr):
    lin_reg = LinearRegression().fit(patches_lr, patches_hr)
    return lin_reg.score(patches_lr, patches_hr)

def train_regression_tree(patches_lr, patches_hr):
    reg_tree = tree.DecisionTreeRegressor().fit(patches_lr, patches_hr)
    return reg_tree.score(patches_lr, patches_hr)

def train_random_forest(patches_lr, patches_hr):
    ran_forest = RandomForestRegressor(n_estimators=10).fit(patches_lr, patches_hr)
    return ran_forest.score(patches_lr, patches_hr)
    
patches_lr, patches_hr = load_training_data()
print("Linear regression:", train_linear_regression(patches_lr, patches_hr))
print("Regression tree:", train_regression_tree(patches_lr, patches_hr))
print("Random forest:", train_random_forest(patches_lr, patches_hr))
