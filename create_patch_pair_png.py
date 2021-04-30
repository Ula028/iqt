import numpy as np
from dipy.data import get_sphere
from dipy.reconst.dti import color_fa, fractional_anisotropy
from dipy.viz import actor, window
import utils

n = 2
m = 2
edge = 2*n + 1

patches_lr, patches_hr = utils.load_training_data(50)

# low resolution patch
s = patches_lr.shape
index = np.random.choice(s[0])
print(index)
patch = patches_lr[index, :]
patch = np.reshape(patch, (edge, edge, edge, 6))
new_patch = np.zeros((edge, edge, edge, 3, 3))
for xc, plane in enumerate(patch):
    for yc, row in enumerate(plane):
        for zc, voxel in enumerate(row):
            new_patch[xc, yc, zc] = utils.restore_duplicates(patch[xc, yc, zc])

# get eigenvalues and eigenvectors
evals, evecs = np.linalg.eigh(new_patch[:, :, :], UPLO="U")
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

path = 'images/patches/patch_lr' + str(index) + '.png'
window.record(scene, n_frames=1, out_path=path,
                size=(1200, 1200))


# high resolution patch
s = patches_hr.shape
patch = patches_hr[index, :]
patch = np.reshape(patch, (m, m, m, 6))
new_patch = np.zeros((m, m, m, 3, 3))
for xc, plane in enumerate(patch):
    for yc, row in enumerate(plane):
        for zc, voxel in enumerate(row):
            new_patch[xc, yc, zc] = utils.restore_duplicates(patch[xc, yc, zc])

# get eigenvalues and eigenvectors
evals, evecs = np.linalg.eigh(new_patch[:, :, :], UPLO="U")
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

path = 'images/patches/patch_hr' + str(index) + '.png'
window.record(scene, n_frames=1, out_path=path,
                size=(1200, 1200))