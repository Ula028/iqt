import numpy as np
from sklearn.metrics import mean_squared_error

import utils
from utils import load_rand_forest_model


subjects_test = ["175136", "180230", "468050",
                 "902242", "886674", "962058", "103212", "792867"]


def subject_dt_rmse(subject, model_name, rate):

    # load previously fitted hr DTIs
    tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")
    tensors_hr = tensor_file_hr['tensors_hr']

    # load tensor file
    tensor_file_rec = np.load(
        'reconstructed/' + subject + model_name +  str(rate) + '_tensors.npz')
    
    # load DTIs and mask from file
    if model_name == 'inter':
        tensors_rec = tensor_file_rec['tensors_hr']
        mask = tensor_file_rec['mask_hr']
    else:
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

    # calculate dt_rmse for a subject
    squared_diff = np.square(tensors_rec[:, mask] - tensors_hr[:, mask])
    summed = np.sum(squared_diff, axis=0)
    square_root = np.sqrt(summed)
    median = np.median(square_root)
    print("Median for subject " + subject + ": " + str(median))
    
    # calculate error as a percentage
    abs_diff = np.abs(tensors_rec[:, mask] - tensors_hr[:, mask])
    perc = np.divide(abs_diff, tensors_hr[:, mask])
    print(np.median(perc))
    
    return median

def total_dt_rmse(model_name, rate):
    results = []
    for subject in subjects_test:
        results.append(subject_dt_rmse(subject, model_name, rate))
    
    return np.mean(results), np.std(results)

if __name__ == "__main__":

    rate = 10
    model_name = 'lin_reg'  # inter, lin_reg, reg_tree, ran_forest
    print(total_dt_rmse(model_name, rate))