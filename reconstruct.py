from operator import sub
import pickle
import timeit

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import utils

upsample_rate = 2  # the super-resolution factor (m in paper)
# the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
input_radius = 2

subjects_test = ["175136", "180230", "468050",
                 "902242", "886674", "962058", "103212", "792867"]


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


def reconstruct(subject, all_indices, tensors_lr, mask_lr, target_res, model, model_name, rate, imputer_name):
    n = input_radius
    m = upsample_rate
    min_present = 0.7
    min_input_size = int(np.floor((2*n + 1) ** 3 * min_present))
    count = 0

    # load models for boundary reconstruction
    if rec_boundary:
        if imputer_name == 'iterative':
            print("Training the iterative imputer...")
            imputer_patches = utils.load_patches(subject)
            imputer = IterativeImputer(max_iter=20, random_state=0)
            imputer = imputer.fit(imputer_patches)
            imputer_patches = 0
        elif imputer_name == 'knn':
            print("Training the knn imputer...")
            imputer_patches = utils.load_patches(subject)
            imputer = KNNImputer()
            imputer = imputer.fit(imputer_patches)
            imputer_patches = 0
        else:
            mean = utils.load_mean()
            covariance = utils.load_covariance()

    all_predictions = np.zeros(target_res)
    dim1, dim2, dim3, dim4, dim5 = target_res
    predictions_mask = np.full((dim1, dim2, dim3), False)

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
        elem_present = np.count_nonzero(p_mask)

        # print("Nonzero:", np.count_nonzero(p_mask))
        # use the patch if it is fully contained in the brain
        if np.all(p_mask):
            patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
            to_predict = True

        # patch is partially contained in the brain and boundary reconstruction is on
        elif rec_boundary and p_mask[n, n, n] and elem_present >= min_input_size:
            print("in boudnary")
            if imputer_name == 'iterative' or imputer_name == 'knn':
                # fill the patch using the imputer
                p_patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
                patch = utils.complete_patch_imputer(p_mask, p_patch, imputer)
            else:
                # fill the patch with conditional mean
                p_patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
                patch = utils.complete_patch_mean(
                    p_mask, p_patch, mean, covariance)
            to_predict = True
            count += 1

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

    # save the reconstructed DTIs
    filename = 'reconstructed/' + subject + model_name + str(rate) + '_tensors.npz'
    np.savez_compressed(filename, tensors_rec=image, mask_rec=predictions_mask)

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
    rmse = mean_squared_error(original_hr.flatten(),
                              reconst_hr.flatten(), squared=False)
    return rmse


if __name__ == "__main__":
    
    rec_boundary = False  # use boundary reconstruction
    imputer_name = 'conditional'  # iterative, conditional
    model_name = 'lin_reg'  # reg_tree, ran_forest, lin_reg
    rate = 10

    starttime = timeit.default_timer()

    # load the model
    if model_name == 'lin_reg':
        model = utils.load_linear_model(rate)
    elif model_name == 'reg_tree':
        model = utils.load_reg_tree_model(rate)
    else:
        model = utils.load_rand_forest_model(rate)

    for subject in tqdm(subjects_test):

        print()
        print("RECONSTRUCTION FOR SUBJECT:", subject)

        # load and preprocess subject data
        tensors_lr, mask_lr, tensors_hr = load_subject_data(subject)
        all_indices, lr_patches, target_resolution = preprocess_data(
            tensors_lr)

        # reconstruct the diffusion tensors
        reconstructed_tensors, reconstructed_tensors_mask = reconstruct(subject,
                                                                        all_indices, lr_patches, mask_lr, target_resolution, model, model_name, rate, imputer_name)

        # load previously fitted DTIs
        tensor_file_hr = np.load(
            "preprocessed_data/" + subject + "tensors_hr.npz")
        tensors_hr = tensor_file_hr['tensors_hr']

        # calculate error
        err = masked_rmse(tensors_hr, reconstructed_tensors,
                          reconstructed_tensors_mask)

        print("Boundary reconstruction:", rec_boundary)
        if rec_boundary:
            if imputer_name == 'iterative':
                print("Method: Iterative Imputer")
            elif imputer_name == 'knn':
                print("Method: KNN Imputer")
            else:
                print("Method: Conditional Mean")
        print("RMSE:", err)

    print("Execution time :", timeit.default_timer() - starttime)
