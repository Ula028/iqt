import numpy as np
import utils
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

upsample_rate = 2  # the super-resolution factor (m in paper)
input_radius = 2 # the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)

def load_subject_data(subject):
    tensor_file_hr = np.load(
        "preprocessed_data/" + subject + "tensors_hr.npz")
    tensor_file_lr = np.load(
        "preprocessed_data/" + subject + "tensors_lr.npz")
    tensors_hr = tensor_file_hr['tensors_hr']
    tensors_lr = tensor_file_lr['tensors_lr']
    mask_lr = tensor_file_lr['mask_lr']
    return tensors_lr, mask_lr, tensors_hr

def preprocess_data(tensors_lr):
    n = input_radius
    m = upsample_rate

    # flatten DT matrices
    s = tensors_lr.shape
    tensors_lr = np.reshape(tensors_lr, (s[0], s[1], s[2], 9))
    
    # calulate the target resolution after upsampling
    target_resolution = (s[0]*m, s[1]*m, s[2]*m, 6)

    # remove duplicate entries to obtain the 6 unique parameters
    tensors_lr = np.delete(tensors_lr, [3, 6, 7], axis=3)
    
    all_indices = utils.create_triples(s[0], s[1], s[2])
    
    # list of central indices for lr patches
    c_indices_lr = []
    
    # extract lr patches
    dims = tensors_lr.shape
    for x in range(n, dims[0] - n):
        for y in range(n, dims[1] - n):
            for z in range(n, dims[2] - n):
                c_indices_lr.append((x, y, z))

    # list of start and end indices for lr
    indices_lr_features = [(x-n, x+n+1, y-n, y+n+1, z-n, z+n+1)
                            for (x, y, z) in c_indices_lr]
    
    n_patches = len(indices_lr_features)
    lr_size = 2*n + 1
    
    lr_patches = np.zeros((n_patches, lr_size, lr_size, lr_size, 6))
    for patch_index, indices in enumerate(indices_lr_features):
        s_x, e_x, s_y, e_y, s_z, e_z = indices
        patch = tensors_lr[s_x:e_x, s_y:e_y, s_z:e_z]
        lr_patches[patch_index] = patch
    
    # flatten lr patches
    vec_len_lr = 6 * lr_size ** 3
    lr_patches = np.reshape(lr_patches, (n_patches, vec_len_lr))
    print(lr_patches.shape)
    
    return all_indices, c_indices_lr, lr_patches, target_resolution
    
def reconstruct(all_indices, c_indices_lr, patches_lr, mask_lr, target_res):
    n = input_radius
    m = upsample_rate
    predictions = []
    # iterate over the low quality image
    print("Reconstructing high quality image...")
    for patch, index in zip(patches_lr, all_indices):
        # use the train model if patch contained in the brain
        if index in c_indices_lr and utils.contained_in_brain(index, n, mask_lr):
            prediction = model.predict(patch.reshape(1, -1))
        # otherwise use the conditional mean
        else:
            # use duplicated central patch for now
            center = np.reshape(patch, (2*n + 1, 2*n + 1, 2*n + 1, 6))[n+1, n+1, n+1]
            copied = np.repeat(center, m * m * m)
            prediction = np.reshape(copied, (1, 48))
        predictions.append(prediction)
    
    reconstructed_img = np.reshape(predictions, target_res)
    reconstructed_img = np.apply_along_axis(utils.restore_duplicates, axis=3, arr=reconstructed_img)
    # reshape the last dimension into diffusion tensors
    reconstructed_img = np.reshape(reconstructed_img, target_res[0], target_res[1], target_res[2], 3, 3)
    
    return reconstructed_img


def load_linear_model():
    with open('linear_model.pickle', 'rb') as handle:
        lin_reg = pickle.load(handle)
    return lin_reg


model = load_linear_model()
tensors_lr, mask_lr, tensors_hr = load_subject_data("962058")
all_indices, c_indices_lr, lr_patches, target_resolution = preprocess_data(tensors_lr)
reconstruction = reconstruct(all_indices, c_indices_lr, lr_patches, mask_lr, target_resolution)
print(reconstruction.shape)
print(tensors_hr.shape)
rmse = mean_squared_error(tensors_hr, reconstruction, squared=False)
print("Score:", rmse)
