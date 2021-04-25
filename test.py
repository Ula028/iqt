import numpy as np
from sklearn.metrics import mean_squared_error

import utils
from utils import load_rand_forest_model


def load_testing_data():
    dict_data = np.load('preprocessed_data/test_data.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Test patches_lr shape:", patches_lr.shape)
    print("Test patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr


test_lr, test_hr = load_testing_data()
ran_forest = utils.load_rand_forest_model()

prediction = ran_forest.predict(test_lr)
rmse = mean_squared_error(test_hr, prediction, squared=False)
print("Score:", rmse)
