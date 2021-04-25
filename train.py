"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def load_training_data():
    dict_data = np.load('preprocessed_data/train_data.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Train patches_lr shape:", patches_lr.shape)
    print("Train patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr


print("Loading data...")
train_lr, train_hr = load_training_data()

# print("Training the imputer...")
# imputer = IterativeImputer(max_iter=10, random_state=0)
# imputer = imputer.fit(train_lr)

print("Training the model...")
# lin_reg = LinearRegression().fit(train_lr, train_hr)
# reg_tree = DecisionTreeRegressor(criterion='mse').fit(train_lr, train_hr)
ran_forest = RandomForestRegressor(
    n_estimators=8, max_depth=20, max_features=26).fit(train_lr, train_hr)

# print("Calculating normal distrubution...")
# mean = np.mean(train_lr, axis=0)
# covariance = np.cov(train_lr, rowvar=False)

# save the model
with open('models/ran_forest_model.pickle', 'wb') as handle:
    pickle.dump(ran_forest, handle)

# # save the normal distribution
# with open('models/mean.pickle', 'wb') as handle:
#     pickle.dump(mean, handle)

# with open('models/covariance.pickle', 'wb') as handle:
#     pickle.dump(covariance, handle)

# # save the imputer
# with open('models/imputer.pickle', 'wb') as handle:
#     pickle.dump(imputer, handle)
