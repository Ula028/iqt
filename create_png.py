import numpy as np
from dipy.data import get_sphere
from dipy.reconst.dti import color_fa, fractional_anisotropy
from dipy.viz import actor, window

from reconstruct import reconstruct

if __name__ == "__main__":

    subject = '103212'
    which = 'lr'

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
        tensor_file_inter = np.load(
            "reconstructed/" + subject + "inter_tensors.npz")

        # get original eigenvalues and eigenvectors
        evals = tensor_file_inter['evals_hr']
        evecs = tensor_file_inter['evecs_hr']

    elif which == 'lin_reg':
        # load reconstructed DTIs
        file = np.load('reconstructed/' + subject + 'lin_reg_tensors.npz', 'rb')
        reconstruction = file['tensors_rec']

        # get reconstructed hr eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(reconstruction[:, :, :], UPLO="U")
        evals = evals[:, :, :, ::-1]
        evecs = evecs[:, :, :, :, ::-1]

    elif which == 'ran_forest':
        # load reconstructed DTIs
        file = np.load('reconstructed/' + subject + 'ran_forest_tensors.npz', 'rb')
        reconstruction = file['tensors_rec']

        # get reconstructed hr eigenvalues and eigenvectors
        evals, evecs = np.linalg.eigh(reconstruction[:, :, :], UPLO="U")
        evals = evals[:, :, :, ::-1]
        evecs = evecs[:, :, :, :, ::-1]

    # create png image
    
    # rotate image
    # evals = np.rot90(evals, k=1, axes=(0, 2))
    # evecs = np.rot90(evecs, k=1, axes=(0, 2))
    
    FA = fractional_anisotropy(evals)

    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs)

    sphere = get_sphere('repulsion724')

    scene = window.Scene()

    cfa = RGB
    cfa /= cfa.max()

    print("Creating png...")
    scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                                  scale=0.3))

    path = 'images/' + subject + 'tensor_ellipsoids_' + which + '.png'
    print("Saving png: " + path)
     
    window.record(scene, n_frames=1, out_path=path,
                  size=(1200, 1200))
