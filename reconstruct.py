import numpy as np
import utils
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

upsample_rate = 2  # the super-resolution factor (m in paper)
# the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
input_radius = 2


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

    # list of central indices of full patches
    c_indices_lr = []

    # extract full lr patches (but not necessarily contained in the brain)
    for x in range(n, s[0] - n):
        for y in range(n, s[1] - n):
            for z in range(n, s[2] - n):
                c_indices_lr.append((x, y, z))

    return all_indices, c_indices_lr, tensors_lr, target_resolution


def reconstruct(all_indices, tensors_lr, mask_lr, target_res, model, mean, covariance):
    n = input_radius
    m = upsample_rate
    
    all_predictions = np.zeros(target_res)
    
    print("tensors_lr before padding:", tensors_lr.shape)
    
    # pad the arrays to avoid going out of bounds and complete partial patches
    tensors_lr = np.pad(tensors_lr, n, mode='constant', constant_values=0)
    tensors_lr = tensors_lr[:, :, :, n:-n]
    mask_lr = np.pad(mask_lr, n, mode='constant', constant_values=False)
    
    print("tensors_lr after padding:", tensors_lr.shape)
    
    # iterate over the low quality image
    print("Reconstructing high quality image...")
    for index in tqdm(all_indices):
        x, y, z = tuple(np.array(index) + n)
        to_predict = False
        
        p_mask = mask_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
        # use the patch if it is contained in the brain
        if np.all(p_mask):
            patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
            to_predict = True    
            
        # use the patch filled with conditional mean if it is partially contained in the brain
        elif np.any(p_mask):
            p_patch = tensors_lr[(x-n):(x+n+1), (y-n):(y+n+1), (z-n):(z+n+1)]
            patch = utils.complete_patch(p_mask, p_patch, mean, covariance)
            patch = np.reshape(patch, (2*n + 1, 2*n + 1, 2*n + 1, 6))  
            to_predict = False
        
        if to_predict:
            patch = patch.flatten().reshape(1, -1)
            prediction = model.predict(patch)
            prediction = np.reshape(prediction, (m, m, m, 6))
            
            for xc, plane in enumerate(prediction):
                for yc, row in enumerate(plane):
                    for zc, voxel in enumerate(row):
                        voxel = utils.restore_duplicates(voxel)
                        all_predictions[m * x + xc, m * y + yc, m * z + zc] = voxel
        
    
    print("Predictions shape:", all_predictions.shape)
    print("Target shape:", target_res)
    image = np.reshape(all_predictions, target_res)

    return image


def load_linear_model():
    with open('models/linear_model.pickle', 'rb') as handle:
        lin_reg = pickle.load(handle)
    return lin_reg

def load_reg_tree_model():
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

# load the model
model = load_linear_model()
mean = load_mean()
covariance = load_covariance()

# load and preprocess subject data
tensors_lr, mask_lr, tensors_hr = load_subject_data("962058")
print("Mask shape:", mask_lr.shape)
all_indices, c_indices_lr, lr_patches, target_resolution = preprocess_data(
    tensors_lr)

# reconstruct the diffusion tensors
reconstructed_tensors = reconstruct(
    all_indices, lr_patches, mask_lr, target_resolution, model, mean, covariance)

print(reconstructed_tensors.shape)
print(tensors_hr.shape)

# save the image
with open('reconstructed_tensors.pickle', 'wb') as handle:
    pickle.dump(reconstructed_tensors, handle)

# load previously fitted DTIs
tensor_file_hr = np.load("preprocessed_data/" + "962058" + "tensors_hr.npz")
tensors_hr = tensor_file_hr['tensors_hr']

# load reconstructed DTIs
with open('reconstructed_tensors.pickle', 'rb') as handle:
    reconstructed_tensors = pickle.load(handle)

# cast to common size if sizes different
new_size = reconstructed_tensors.shape
if new_size != tensors_hr.shape:
    tensors_hr = tensors_hr[:new_size[0], :new_size[1], :new_size[2]]
        
rmse = mean_squared_error(tensors_hr.flatten(), reconstructed_tensors.flatten(), squared=False)
print("RMSE:", rmse)

