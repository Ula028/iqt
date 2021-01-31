"""A script that creates training data for IQT random forest
training from a typical HCP dataset.
"""
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.align.reslice import reslice
from numpy import savez_compressed
import dipy.reconst.dti as dti

# settings
path = "data\\100307_3T_Diffusion_preproc\\100307\\T1w\\Diffusion\\"
dw_file = path + "data.nii.gz"
bvals_file = path + "bvals"
bvecs_file = path + "bvecs"
# not used (median_otsu used instead)
mask_file = path + "nodif_brain_mask.nii.gz"
grad_file = path + "grad_dev.nii.gz"

upsample_rate = 2  # the super-resolution factor (m in paper)
# the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
input_radius = 2
datasample_rate = 2  # determines the size of training sets. From each subject, we randomly draw patches with probability 1/datasample_rate
no_rnds = 8  # no of separate training sets to be created

"""
Computes DTIs on the original DWIs and its downsampled version.
As a result, we obtain high-res and low-res DTIs.
"""


def compute_dti_respairs(dw_file, bvals_file, bvecs_file):
    # read in the DWI data
    data_hr, affine_hr, voxsize = load_nifti(dw_file, return_voxsize=True)
    # read bvals and bvecs text files
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs)
    # get the brain mask for high-res DWI
    maskdata_hr, mask_hr = median_otsu(data_hr, vol_idx=range(10, 50), median_radius=3,
                                       numpass=1, autocrop=False, dilate=2)

    # compute DTI for the original high-res DWI
    tenmodel = dti.TensorModel(gtab)
    print("Fitting high resolution DTs...")
    tenfit_hr = tenmodel.fit(maskdata_hr)
    quadratic_tensors_hr = tenfit_hr.quadratic_form
    print("High resolutions tensors:", quadratic_tensors_hr.shape)

    # downsample the DWI
    data_lr, affine_lr = reslice(data_hr, affine_hr, voxsize, [
                                 i*upsample_rate for i in voxsize])
    # get the brain mask for low-res DWI
    maskdata_lr, mask_lr = median_otsu(data_lr, vol_idx=range(10, 50), median_radius=3,
                                       numpass=1, autocrop=False, dilate=2)
    # compute DTI for the downsampled DWI
    print("Fitting low resolution DTs...")
    tenfit_lr = tenmodel.fit(maskdata_lr)
    quadratic_tensors_lr = tenfit_lr.quadratic_form
    print("Low resolutions tensors:", quadratic_tensors_lr.shape)

    # save DTIs and masks to a file
    savez_compressed('tensors.npz', tensors_hr=quadratic_tensors_hr, mask_hr=mask_hr,
                     tensors_lr=quadratic_tensors_lr, mask_lr=mask_lr)


"""
Extracts patch-pairs from low/high resolution DTIs
to create an exhaustive list of all valid patch-pairs and stores them
in a large matrix.
"""


def compute_patchlib(input_radius, datasample_rate):
    n = input_radius
    tensor_file = np.load('tensors.npz')
    mask_lr = tensor_file['mask_lr']

    print("Computing locations of valid patch pairs...")

    # list of central indices for lr
    indices_lr_features = []

    dims = mask_lr.shape
    for x in range(n, dims[0] - n):
        for y in range(n, dims[1] - n):
            for z in range(n, dims[2] - n):

                if mask_lr[x + n, y + n, z + n] and \
                        mask_lr[x + n, y + n, z - n] and \
                        mask_lr[x + n, y - n, z + n] and \
                        mask_lr[x + n, y - n, z - n] and \
                        mask_lr[x - n, y + n, z + n] and \
                        mask_lr[x - n, y + n, z - n] and \
                        mask_lr[x - n, y - n, z + n] and \
                        mask_lr[x - n, y - n, z - n]:
                    indices_lr_features.append((x, y, z))

    # list of corner indices for hr
    indices_hr_features = [(x*n, y*n, z*n)
                           for (x, y, z) in indices_lr_features]

    tensors_hr = tensor_file['tensors_hr']
    tensors_lr = tensor_file['tensors_lr']


compute_patchlib(input_radius, datasample_rate)
