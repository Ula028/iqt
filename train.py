"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
import seaborn as sns



def load_training_data():
    dict_data = np.load('preprocessed_data/train_data.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Train patches_lr shape:", patches_lr.shape)
    print("Train patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr


def load_testing_data():
    dict_data = np.load('preprocessed_data/test_data.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Test patches_lr shape:", patches_lr.shape)
    print("Test patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr


print("Loading data...")
train_lr, train_hr = load_training_data()
test_lr, test_hr = load_testing_data()

# print("Training the imputer...")
# imputer = IterativeImputer(max_iter=10, random_state=0)
# imputer = imputer.fit(train_lr)

# print("Training the model...")
# lin_reg = LinearRegression().fit(train_lr, train_hr)
# reg_tree = DecisionTreeRegressor(criterion='mse').fit(train_lr, train_hr)
# ran_forest = RandomForestRegressor(n_estimators=10).fit(train_lr, train_hr)

# grid search for RandomForestRegressor
print("Performing grid search...")
param_grid = [
    {'n_estimators': [10, 50, 100],
     'max_depth': [16, 32, 64], 
     'n_jobs': [-1]}, 
    {'max_features': ['sqrt', 'log2'], 
     'bootstrap': [True, False], 
     'n_jobs': [-1]}
]
rand_forest = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rand_forest, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, refit=True, verbose=3)
grid_search.fit(train_lr, train_hr)

# get the best model
model = grid_search.best_estimator_

# print the results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# print("Calculating normal distrubution...")
# mean = np.mean(train_lr, axis=0)
# covariance = np.cov(train_lr, rowvar=False)

# prediction = ran_forest.predict(test_lr)
# rmse = mean_squared_error(test_hr, prediction, squared=False)
# print("Score:", rmse)

# save the model
with open('models/reg_tree_model.pickle', 'wb') as handle:
    pickle.dump(model, handle)

# # save the normal distribution
# with open('models/mean.pickle', 'wb') as handle:
#     pickle.dump(mean, handle)

# with open('models/covariance.pickle', 'wb') as handle:
#     pickle.dump(covariance, handle)

# # save the imputer
# with open('models/imputer.pickle', 'wb') as handle:
#     pickle.dump(imputer, handle)