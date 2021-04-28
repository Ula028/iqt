import numpy as np
from sklearn.metrics import mean_squared_error

import utils
from utils import load_rand_forest_model


subjects_test = ["175136", "180230", "468050",
                 "902242", "886674", "962058", "103212", "792867"]

model_name = 'lin_reg'  # cubic interpolation, lin_reg, reg_tree, ran_forest


def subject_dt_rmse(subject, model_name):

    # load previously fitted hr DTIs
    tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")
    tensors_hr = tensor_file_hr['tensors_hr']

    # load reconstructed hr DTIs
    tensor_file_rec = np.load(
        'reconstructed/' + subject + model_name + '_tensors.npz')
    tensors_rec = tensor_file_rec['tensors_rec']
    mask = tensor_file_rec['mask_rec']

    # cast to common size if sizes different
    new_size = tensors_rec.shape
    if new_size != tensors_hr.shape:
        tensors_hr = tensors_hr[:new_size[0], :new_size[1], :new_size[2]]


    # flatten DT matrices
    tensors_hr = np.reshape(
        tensors_hr, (new_size[0], new_size[1], new_size[2], 9))
    tensors_rec = np.reshape(
        tensors_rec, (new_size[0], new_size[1], new_size[2], 9))

    # remove duplicate entries to obtain the 6 unique parameters
    tensors_hr = np.delete(tensors_hr, [3, 6, 7], axis=3)
    tensors_rec = np.delete(tensors_rec, [3, 6, 7], axis=3)
    
    # reshape to obtain vector of errors
    tensors_hr = np.reshape(tensors_hr, (new_size[0]*new_size[1]*new_size[2], 6)).T
    tensors_rec = np.reshape(tensors_rec, (new_size[0]*new_size[1]*new_size[2], 6)).T
    mask = mask.flatten()

    rmse = mean_squared_error(tensors_hr[:, mask],
                              tensors_rec[:, mask], squared=False, multioutput='raw_values')
    dt_rmse = np.median(rmse)
    
    return dt_rmse

def total_dt_rmse(model_name):
    results = []
    for subject in subjects_test:
        results.append(subject_dt_rmse(subject, model_name))
    
    return np.mean(results), np.std(results)

if __name__ == "__main__":

    print(total_dt_rmse(model_name))