from dipy.viz import window, actor
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, color_fa

import numpy as np
import pickle

subject = "962058"

# load previously fitted DTIs
tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")

# load reconstructed DTIs
with open('reconstructed_tensors.pickle', 'rb') as handle:
    reconstruction = pickle.load(handle)

# get original hr eigenvalues and eigenvectors
evals_hr = tensor_file_hr['evals_hr']
evecs_hr = tensor_file_hr['evecs_hr']

# get reconstructed hr eigenvalues and eigenvectors
evals_rec, evecs_rec = np.linalg.eigh(reconstruction[:, :, :], UPLO="U")

evals_rec = evals_rec[:, :, :, ::-1]
evecs_rec = evecs_rec[:, :, :, :, ::-1]

# # hr
# FA_hr = fractional_anisotropy(evals_hr)
# FA_hr[np.isnan(FA_hr)] = 0

# FA_hr = np.clip(FA_hr, 0, 1)
# RGB_hr = color_fa(FA_hr, evecs_hr)


# reconstructed hr
FA_rec = fractional_anisotropy(evals_rec)

FA_rec = np.clip(FA_rec, 0, 1)
RGB_rec = color_fa(FA_rec, evecs_rec)

sphere = get_sphere('repulsion724')

scene = window.Scene()

cfa_rec = RGB_rec
cfa_rec /= cfa_rec.max()

scene.add(actor.tensor_slicer(evals_rec, evecs_rec, scalar_colors=cfa_rec, sphere=sphere,
                              scale=0.05))

window.record(scene, n_frames=1, out_path='tensor_ellipsoids_rec.png',
              size=(1200, 1200))
