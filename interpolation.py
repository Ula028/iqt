import dipy.reconst.dti as dti
import numpy as np
from dipy.align.reslice import reslice
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from sklearn.metrics import mean_squared_error

import utils

# settings
dw_fname = "data.nii.gz"
bvals_fname = "bvals"
bvecs_fname = "bvecs"
grad_file = "grad_dev.nii.gz"
upsample_rate = 2  # the super-resolution factor (m in paper)

subjects_test = ["175136", "180230", "468050",
                 "902242", "886674", "962058", "103212", "792867"]

def linear_interpolation(subject):
    """ Computes DTIs from DWI upsampled by linear interpolation.

    Args:
        subject (string): subject id
    """

    print("\nSUBJECT:", subject)
    # get paths
    path = utils.join_path(subject)
    dw_file = path + dw_fname
    bvals_file = path + bvals_fname
    bvecs_file = path + bvecs_fname

    # read in the DWI data
    data_hr, affine_hr, voxsize = load_nifti(dw_file, return_voxsize=True)
    # read bvals and bvecs text files
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs)

    # downsample the DWI
    print("Downsampling the DWI...")
    downsampled_voxsize = [i*upsample_rate for i in voxsize]
    data_lr, affine_lr = reslice(
        data_hr, affine_hr, voxsize, downsampled_voxsize)

    # upsample the DWI
    print("Upsampling the DWI...")
    upsampled_voxsize = [i/upsample_rate for i in downsampled_voxsize]
    data_hr, affine_hr = reslice(
        data_lr, affine_lr, downsampled_voxsize, upsampled_voxsize)

    # get the brain mask for high-res DWI
    maskdata_hr, mask_hr = median_otsu(data_hr, vol_idx=range(10, 50), median_radius=3,
                                       numpass=1, autocrop=False, dilate=2)

    # compute DTI for the upsampled high-res DWI
    tenmodel = dti.TensorModel(gtab)

    print("Fitting high resolution DTs...")
    tenfit_hr = tenmodel.fit(maskdata_hr)
    quadratic_tensors_hr = tenfit_hr.quadratic_form
    evals_hr = tenfit_hr.evals
    evecs_hr = tenfit_hr.evecs
    print("High resolutions tensors:", quadratic_tensors_hr.shape)
    # save DTIs, eigenvectors, eigenvalues and mask to a file
    filename = "reconstructed/inter" + subject + "tensors.npz"
    np.savez_compressed(filename, tensors_hr=quadratic_tensors_hr, mask_hr=mask_hr,
                        evals_hr=evals_hr, evecs_hr=evecs_hr)

    return quadratic_tensors_hr


for subject in subjects_test:
    tensors_interpolated = linear_interpolation(subject)

# load previously fitted DTIs
tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")
tensors_hr = tensor_file_hr['tensors_hr']

# cast to common size if sizes different
new_size = tensors_interpolated.shape
if new_size != tensors_hr.shape:
    original_hr = tensors_hr[:new_size[0], :new_size[1], :new_size[2]]

# calculate rmse
rmse = mean_squared_error(original_hr.flatten(),
                          tensors_interpolated.flatten(), squared=False)
