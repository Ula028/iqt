
import numpy as np
import pickle
from sklearn.impute import KNNImputer


def join_path(subject):
    """Returns the path for diffusion data for a particular subject.

    Args:
        subject (string): subject id

    Returns:
        string: the path string for this subject
    """
    return "raw_data/" + subject + "_3T_Diffusion_preproc/" + subject + "/T1w/Diffusion/"


def restore_duplicates(tensor):
    """Creates a flattened diffusion tensor from 6 unique diffusion parameters

    Args:
        tensor ([double]): array containing 6 diffusion parameters

    Returns:
        [double]: a diffusion tensor with 9 elements
    """
    d_yx = tensor[1]
    d_zx = tensor[2]
    d_zy = tensor[4]
    new_tensor = np.array([[tensor[0], tensor[1], tensor[2]],
                           [d_yx, tensor[3], tensor[4]],
                           [d_zx, d_zy, tensor[5]]])

    return new_tensor


def create_triples(x_max, y_max, z_max):
    """Generate all possible coordinates in a given 3D space starting from 0

    Args:
        x_max (int): maximum x coordinate
        y_max (int): maximum y coordinate
        z_max (int): maximum z coordinate

    Returns:
        [(int, int, int)]: list of triples
    """
    triples = []
    for x in range(0, x_max):
        for y in range(0, y_max):
            for z in range(0, z_max):
                triples.append((x, y, z))
    return triples


def complete_patch_mean(p_mask, p_patch, mean, covariance):
    # calculate the conditional mean given a subset of components
    p_patch = p_patch.flatten()
    p_mask = np.repeat(p_mask[:, :, :, np.newaxis], 6, axis=3)
    p_mask = p_mask.flatten()
    missing_idx = p_mask == False
    known_idx = p_mask == True

    u1 = mean[missing_idx]
    u2 = mean[known_idx]
    cov12 = covariance[missing_idx, :][:, known_idx]
    cov22 = covariance[known_idx, :][:, known_idx]
    # cov22_inv = np.linalg.inv(covariance[known_idx, :][:, known_idx])
    diff = p_patch[known_idx] - u2
    x = np.linalg.solve(cov22, diff)

    # cond_mean = u1 + np.linalg.multi_dot([cov12, cov22_inv, diff])
    cond_mean = u1 + np.dot(cov12, x)

    # insert the conditional mean
    p_patch[missing_idx] = cond_mean
    return p_patch


def complete_patch_imputer(p_mask, p_patch, imputer):
    p_patch = p_patch.reshape(1, 750)
    p_mask = np.repeat(p_mask[:, :, :, np.newaxis], 6, axis=3)
    p_mask = p_mask.reshape(1, 750)
    missing_idx = p_mask == False

    p_patch[missing_idx] = np.nan
    patch = imputer.transform(p_patch)

    return patch


def load_linear_model(datasample_rate):
    with open('models/lin_reg_model' + str(datasample_rate) + '.pickle', 'rb') as handle:
        lin_reg = pickle.load(handle)
    return lin_reg


def load_reg_tree_model(datasample_rate):
    with open('models/reg_tree_model' + str(datasample_rate) + '.pickle', 'rb') as handle:
        reg_tree = pickle.load(handle)
    return reg_tree


def load_rand_forest_model(datasample_rate):
    path = 'models/ran_forest_model' + str(datasample_rate) + '.pickle'
    with open(path, 'rb') as handle:
        reg_forest = pickle.load(handle)
    return reg_forest


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

def load_patches(subject, hr=False):
    dict_data = np.load('preprocessed_data/' + subject + 'patches.npz')
    if hr:
        patches = dict_data['patches_hr']
    else:
        patches = dict_data['patches_lr']
    return patches

def load_testing_data(datasample_rate):
    print("Loading testing data...")
    dict_data = np.load('preprocessed_data/test_data' + str(datasample_rate) + '.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Test patches_lr shape:", patches_lr.shape)
    print("Test patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr

def load_training_data(datasample_rate):
    print("Loading training data...")
    dict_data = np.load('preprocessed_data/train_data' + str(datasample_rate) + '.npz')
    patches_lr = dict_data['patches_lr']
    patches_hr = dict_data['patches_hr']
    print("Train patches_lr shape:", patches_lr.shape)
    print("Train patches_hr shape:", patches_hr.shape)
    return patches_lr, patches_hr

def load_scaler():
    with open('models/min_max_scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    return scaler