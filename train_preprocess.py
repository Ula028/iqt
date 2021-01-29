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
mask_file = path + "nodif_brain_mask.nii.gz" # not used (median_otsu used instead)
grad_file = path + "grad_dev.nii.gz"

upsample_rate = 2 # the super-resolution factor (m in paper)
input_radius = 2 # the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
datasample_rate = 2 # determines the size of training sets. From each subject, we randomly draw patches with probability 1/datasample_rate
no_rnds = 8 # no of separate training sets to be created

def compute_dti_respairs(dw_file, bvals_file, bvecs_file):
    # read in the DWI data
    data_hr, affine_hr, voxsize = load_nifti(dw_file, return_voxsize=True)
    # read bvals and bvecs text files
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs)
    # get the brain mask
    maskdata_hr, mask_hr = median_otsu(data_hr, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)
    
    # compute DTI for the original high-res DWI
    tenmodel = dti.TensorModel(gtab)
    tenfit_hr = tenmodel.fit(maskdata_hr)
    print(tenfit_hr.quadratic_form.shape)
    savez_compressed('tensors_hr.npz', tenfit_hr.quadratic_form)
    
    # downsample the DWI
    data_lr, affine_lr = reslice(data_hr, affine_hr, voxsize, [i*datasample_rate for i in voxsize])
    # get the brain mask
    maskdata_lr, mask_lr = median_otsu(data_lr, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)
    # compute DTI for the downsampled DWI
    tenfit_lr = tenmodel.fit(maskdata_lr)
    print(tenfit_lr.quadratic_form.shape)
    savez_compressed('tensors_lr.npz', tenfit_lr.quadratic_form)
    
compute_dti_respairs(dw_file, bvals_file, bvecs_file)