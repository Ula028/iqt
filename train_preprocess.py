"""A script that creates training data for IQT random forest
training from a typical HCP dataset.
"""
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti

# settings
path = "data\\100307_3T_Diffusion_preproc\\100307\\T1w\\Diffusion\\"
dw_file = path + "data.nii.gz"
bvals_file = path + "bvals"
bvecs_file = path + "bvecs"
mask_file = path + "nodif_brain_mask.nii.gz" # median_otsu used mow instead
grad_file = path + "grad_dev.nii.gz"

upsample_rate = 2 # the super-resolution factor (m in paper)
input_radius = 2 # the radius of the low-res input patch i.e. the input is a cubic patch of size (2*input_radius+1)^3 (n in paper)
datasample_rate = 2 # determines the size of training sets. From each subject, we randomly draw patches with probability 1/datasample_rate
no_rnds = 8 # no of separate training sets to be created

def compute_dti_respairs(dw_file, bvals_file, bvecs_file):
    # read in the DWI data
    data, affine, voxsize = load_nifti(dw_file, return_voxsize=True)
    # read bvals and bvecs text files
    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    gtab = gradient_table(bvals, bvecs)
    # read the brain mask
    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)
    print(maskdata.shape)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata[:, :, 51:52])
    
    return tenfit

tenfit = compute_dti_respairs(dw_file, bvals_file, bvecs_file)

from dipy.reconst.dti import fractional_anisotropy, color_fa

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)

from dipy.data import get_sphere
sphere = get_sphere('repulsion724')

from dipy.viz import window, actor

scene = window.Scene()

# evals = tenfit.evals[13:43, 44:74, 28:29]
# evecs = tenfit.evecs[13:43, 44:74, 28:29]
evals = tenfit.evals
evecs = tenfit.evecs

cfa = RGB
cfa /= cfa.max()

scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                              scale=0.3))

window.record(scene, n_frames=1, out_path='tensor_ellipsoids.png',
              size=(600, 600))