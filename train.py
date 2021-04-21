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

# imputer = IterativeImputer(max_iter=10, random_state=0)
# imputer = imputer.fit(train_lr[:300, :])

print("Training the model...")
# lin_reg = LinearRegression().fit(train_lr, train_hr)
# reg_tree = DecisionTreeRegressor(criterion='mse').fit(train_lr, train_hr)
ran_forest = RandomForestRegressor(n_estimators=10).fit(train_lr, train_hr)

# print("Calculating normal distrubution...")
# mean = np.mean(train_lr, axis=0)
# covariance = np.cov(train_lr, rowvar=False)

prediction = ran_forest.predict(test_lr)
rmse = mean_squared_error(test_hr, prediction, squared=False)
print("Score:", rmse)

# save the model
with open('models/ran_forest.pickle', 'wb') as handle:
    pickle.dump(ran_forest, handle)

# # save the normal distribution
# with open('models/mean.pickle', 'wb') as handle:
#     pickle.dump(mean, handle)

# with open('models/covariance.pickle', 'wb') as handle:
#     pickle.dump(covariance, handle)

# # save the imputer
# with open('models/imputer.pickle', 'wb') as handle:
#     pickle.dump(imputer, handle)