"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle


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


train_lr, train_hr = load_training_data()
test_lr, test_hr = load_testing_data()

print("Training decision tree model...")
# lin_reg = LinearRegression().fit(train_lr, train_hr)
reg_tree = DecisionTreeRegressor().fit(train_lr, train_hr)
# ran_forest = RandomForestRegressor(n_estimators=10).fit(train_lr, train_hr)

print("Calculating normal distrubution...")
mean = np.mean(train_lr, axis=0)
covariance = np.cov(train_lr, rowvar=False)

prediction = reg_tree.predict(test_lr)
rmse = mean_squared_error(test_hr, prediction, squared=False)
print("Score:", rmse)


# save the model
with open('models/reg_tree_model.pickle', 'wb') as handle:
    pickle.dump(reg_tree, handle)

# save the normal distribution
with open('models/mean.pickle', 'wb') as handle:
    pickle.dump(mean, handle)

with open('models/covariance.pickle', 'wb') as handle:
    pickle.dump(covariance, handle)