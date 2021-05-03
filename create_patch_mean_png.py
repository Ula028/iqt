import numpy as np
from dipy.data import get_sphere
from dipy.reconst.dti import color_fa, fractional_anisotropy
from dipy.viz import actor, window
import utils

n = 2
m = 2
edge = 2*n + 1

# [[True,  True,  True,  False,  True],
#          [True,  True,  True,  False,  True],
#          [True,  True,  True,  True, True],
#          [True,  True, True, True, False],
#          [True, True, False, False, False]],

        # [[True,  True,  False,  False,  False],
        #  [True,  True,  False,  False,  False],
        #  [True,  True,  True,  False, False],
        #  [True,  True, True, True, False],
        #  [True, True, True, True, True]],


mask = np.array([[[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False]],

        [[False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False]],

        [[False,  False,  False,  False,  False],
         [True,  False,  False,  False,  False],
         [True,  True,  False,  False, False],
         [True,  True, True, False, False],
         [True, True, True, True, False]],

        [[True,  True,  True,  True,  True],
         [True,  True,  True,  True,  True],
         [True,  True,  True,  True,  True],
         [True,  True,  True,  True,  True],
         [True,  True,  True,  True , True]],

        [[True,  True,  True,  True,  True],
         [True,  True,  True,  True,  True],
         [True, True,  True,  True,  True],
         [True, True,  True,  True,  True],
         [True,  True,  True,  True,  True]]])

patches_lr, patches_hr = utils.load_training_data(50)
patches_hr = 0
print(mask.shape)

# FULL low resolution patch
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

path = 'images/patches/' + str(index) + 'patch_full.png'
window.record(scene, n_frames=1, out_path=path,
              size=(1200, 1200))


# COMPLETED low resolution patch
s = patches_lr.shape
patch = patches_lr[index, :]
patch = np.reshape(patch, (edge, edge, edge, 6))

mean = utils.load_mean()
covariance = utils.load_covariance()

patch = utils.complete_patch_mean(mask, patch, mean, covariance)
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

path = 'images/patches/' + str(index) + 'patch_completed.png'
window.record(scene, n_frames=1, out_path=path,
              size=(1200, 1200))
