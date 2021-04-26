import numpy as np
from sklearn.metrics import mean_squared_error

import utils
from utils import load_rand_forest_model

test_lr, test_hr = utils.load_testing_data()
ran_forest = utils.load_rand_forest_model()

prediction = ran_forest.predict(test_lr)
rmse = mean_squared_error(test_hr, prediction, squared=False)
print("Score:", rmse)
