import numpy as np
from sklearn.metrics import mean_squared_error

import utils
from utils import load_rand_forest_model


if __name__ == "__main__":
    print("Loading testing data...")
    test_lr, test_hr = utils.load_testing_data()
    lin_reg = utils.load_linear_model()

    print("Making pradictions...")
    prediction = lin_reg.predict(test_lr)
    print("Calculating error")
    rmse = mean_squared_error(test_hr, prediction, squared=False)
    print("Score:", rmse)