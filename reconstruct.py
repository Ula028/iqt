import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import timeit

import utils

upsample_rate = 2  # the super-resolution factor (m in paper)
# the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
input_radius = 2
rec_boundary = False  # use boundary reconstruction
use_imputer = False # use KNNImputer to fill missing values in partial patches (otherwise use conditional mean)
model_name = 'linear' # 'reg_tree'

def load_subject_data(subject):
    """Load low resolution tensors, low resolution mask and high resolution tensors of a subject

    Args:
        subject (string): subject id

    Returns:
        ([double], [double], [double]): low resolution tensors, low resolution mask, high resolution tensors
    """
    tensor_file_hr = np.load(
        "preprocessed_data/" + subject + "tensors_hr.npz")
    tensor_file_lr = np.load(
        "preprocessed_data/" + subject + "tensors_lr.npz")
    tensors_hr = tensor_file_hr['tensors_hr']
    tensors_lr = tensor_file_lr['tensors_lr']
    mask_lr = tensor_file_lr['mask_lr']
    return tensors_lr, mask_lr, tensors_hr


def preprocess_data(tensors_lr):
    """Preprocess data for image reconstruction

    Args:
        tensors_lr ([double]): 3D array of low resolution tensors

    Returns:
        ([(int, int, int)], [(int, int, int)], [double], (int, int, int, int)): list of central indices of all patches,
        list of central indices of full patches, an array of preprocessed low resolution tensors,
        target resolution of the image after reconstruction
    """
    n = input_radius
    m = upsample_rate

    # flatten DT matrices
    s = tensors_lr.shape
    print("Low resolution tensors shape:", s)
    tensors_lr = np.reshape(tensors_lr, (s[0], s[1], s[2], 9))

    # the target resolution after upsampling
    target_resolution = (s[0]*m, s[1]*m, s[2]*m, 3, 3)
    print("Target resolution after upsampling:", target_resolution)

    # remove duplicate entries to obtain the 6 unique parameters
    tensors_lr = np.delete(tensors_lr, [3, 6, 7], axis=3)

    all_indices = utils.create_triples(s[0], s[1], s[2])

    return all_indices, tensors_lr, target_resolution


def reconstruct(all_indices, tensors_lr, mask_lr, target_res, model_name, use_imputer):
    n = input_radius
    m = upsample_rate
    
    # load the model
    if model_name == 'linear':
        model = load_linear_model()
    elif model_name == 'reg_tree':
        model = load_reg_tree_model()
    else:
        model = load_rand_forest_model()
        
    # print("Depth of the decision tree:", model.get_depth())
    
    # load models for boundary reconstruction
    if rec_boundary:
        if use_imputer:
            imputer = load_imputer()
        else:
            mean = load_mean()
            covariance = load_covariance()

    all_predictions = np.zeros(target_res)
    predictions_mask = np.zeros(target_res)

    # pad the arrays to avoid going out of bounds and complete partial patches
    tensors_lr = np.pad(tensors_lr, n, mode='constant', constant_values=0)
    tensors_lr = tensors_lr[:, :, :, n:-n]
    mask_lr = np.pad(mask_lr, n, mode='constant', constant_values=False)

    # iterate over the low quality image
    print("Reconstructing high quality image...")
    it_indices = tuple(np.array(all_indices) + n)

    for index in tqdm(it_indices):
        x, y, z = index
        to_predict = False

        p_mask = mask_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]

        # use the patch if it is fully contained in the brain
        if np.all(p_mask):
            patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
            to_predict = True

        # patch is partially contained in the brain and boundary reconstruction is on
        elif rec_boundary and p_mask[n, n, n] == True:
            if use_imputer:
                # fill the patch using k-Nearest Neighbors imputer
                p_patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
                patch = utils.complete_patch_imputer(p_mask, p_patch, imputer)
            else:
                # fill the patch with conditional mean
                p_patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
                patch = utils.complete_patch_mean(p_mask, p_patch, mean, covariance)
            to_predict = True

        if to_predict:
            patch = patch.flatten().reshape(1, -1)
            prediction = model.predict(patch)
            prediction = np.reshape(prediction, (m, m, m, 6))

            for xc, plane in enumerate(prediction):
                for yc, row in enumerate(plane):
                    for zc, voxel in enumerate(row):
                        if p_mask[n, n, n] == True:
                            voxel = utils.restore_duplicates(voxel)
                            all_predictions[m * (x - n) + xc, m *
                                            (y - n) + yc, m * (z - n) + zc] = voxel
                            predictions_mask[m * (x - n) + xc, m *
                                            (y - n) + yc, m * (z - n) + zc] = True

    print("Predictions shape:", all_predictions.shape)
    print("Target shape:", target_res)
    image = np.reshape(all_predictions, target_res)

    return image, predictions_mask

def masked_rmse(original_hr, reconst_hr, reconst_mask):
    
    # cast to common size if sizes different
    new_size = reconst_hr.shape
    if new_size != original_hr.shape:
        original_hr = original_hr[:new_size[0], :new_size[1], :new_size[2]]
    
    # mask original hr tensors to calculate error without boundary
    to_delete = reconst_mask == False
    original_hr[to_delete] = 0
    
    # calculate rmse
    rmse = mean_squared_error(original_hr.flatten(), reconst_hr.flatten(), squared=False)
    return rmse

def load_linear_model():
    with open('models/linear_model.pickle', 'rb') as handle:
        lin_reg = pickle.load(handle)
    return lin_reg


def load_reg_tree_model():
    with open('models/reg_tree_model.pickle', 'rb') as handle:
        reg_tree = pickle.load(handle)
    return reg_tree

def load_rand_forest_model():
    # LOADS LINEAR MODEL
    with open('models/linear_model.pickle', 'rb') as handle:
        reg_tree = pickle.load(handle)
    return reg_tree


def load_mean():
    with open('models/mean.pickle', 'rb') as handle:
        mean = pickle.load(handle)
    return mean


def load_covariance():
    with open('models/covariance.pickle', 'rb') as handle:
        covariance = pickle.load(handle)
    return covariance

def load_imputer():
    with open('models/imputer.pickle', 'rb') as handle:
        imputer = pickle.load(handle)
    return imputer

subject = "175136"

starttime = timeit.default_timer()

# load and preprocess subject data
tensors_lr, mask_lr, tensors_hr = load_subject_data(subject)
all_indices, lr_patches, target_resolution = preprocess_data(
    tensors_lr)

# reconstruct the diffusion tensors
reconstructed_tensors, reconstructed_tensors_mask = reconstruct(
    all_indices, lr_patches, mask_lr, target_resolution, model_name, use_imputer)

# save the reconstructed DTIs
with open('reconstructed_tensors.pickle', 'wb') as handle:
    pickle.dump(reconstructed_tensors, handle)

# load previously fitted DTIs
tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")
tensors_hr = tensor_file_hr['tensors_hr']

# load reconstructed DTIs
with open('reconstructed_tensors.pickle', 'rb') as handle:
    reconstructed_tensors = pickle.load(handle)

# calculate error
err = masked_rmse(tensors_hr, reconstructed_tensors, reconstructed_tensors_mask)

print("Boundary reconstruction:", rec_boundary)
if rec_boundary:
    if use_imputer:
        print("Method: imputer")
    else:
        print("Method: conditional mean")
print("RMSE:", err)
print("Execution time :", timeit.default_timer() - starttime)
