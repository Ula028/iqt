"""A script that trains a model for IQT random forest
using previously created dataset.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
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

print("Training linear regression model...")
lin_reg = LinearRegression().fit(train_lr, train_hr)
prediction = lin_reg.predict(test_lr)
rmse = mean_squared_error(test_hr, prediction, squared=False)
print("Score:", rmse)
# reg_tree = tree.DecisionTreeRegressor().fit(train_lr, train_hr)
# ran_forest = RandomForestRegressor(n_estimators=10).fit(train_lr, train_hr)

# save the model
with open('linear_model.pickle', 'wb') as handle:
    pickle.dump(lin_reg, handle)
