from dipy.viz import window, actor
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, color_fa

import numpy as np
import pickle


if __name__ == "__main__":

    subject = '175136'
    which = 'hr'

    if which == 'hr':
        # load previously fitted DTIs
        tensor_file_hr = np.load(
            "preprocessed_data/" + subject + "tensors_hr.npz")

        # get original hr eigenvalues and eigenvectors
        evals = tensor_file_hr['evals_hr']
        evecs = tensor_file_hr['evecs_hr']
    
    elif which == 'lr':
        # load previously fitted DTIs
        tensor_file_hr = np.load(
            "preprocessed_data/" + subject + "tensors_lr.npz")

        # get original hr eigenvalues and eigenvectors
        evals = tensor_file_hr['evals_lr']
        evecs = tensor_file_hr['evecs_lr']

    elif which == 'interpolation':
        # load interpolated DTIs
        tensor_file_inter = np.load("reconstructed/" + subject + "inter_tensors_hr.npz")

        # get original eigenvalues and eigenvectors
        evals = tensor_file_inter['evals_hr']
        evecs = tensor_file_inter['evecs_hr']


    
    elif which == 'lin_reg':
        # load reconstructed DTIs
        with open('reconstructed/lin_reg' + subject + 'rec_tensors.pickle', 'rb') as handle:
            reconstruction = pickle.load(handle)

        # get reconstructed hr eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(reconstruction[:, :, :], UPLO="U")
        evals = evals[:, :, :, ::-1]
        evecs = evecs[:, :, :, :, ::-1]

    elif which == 'lin_reg':
        # load reconstructed DTIs
        with open('reconstructed/' + subject + 'rec_tensors_hr.pickle', 'rb') as handle:
            reconstruction = pickle.load(handle)

        # get reconstructed hr eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(reconstruction[:, :, :], UPLO="U")
        evals = evals[:, :, :, ::-1]
        evecs = evecs[:, :, :, :, ::-1]

    
    # create png image
    FA = fractional_anisotropy(evals)

    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs)

    sphere = get_sphere('repulsion724')

    scene = window.Scene()

    cfa = RGB
    cfa /= cfa.max()

    scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                                  scale=0.3))

    path = 'images/' + subject + 'tensor_ellipsoids_' + which + '.png'
    window.record(scene, n_frames=1, out_path=path,
                  size=(1200, 1200))
