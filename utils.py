
import numpy as np
from numpy.linalg.linalg import cond

def join_path(subject):
    """Returns the path for diffusion data for a particular subject.

    Args:
        subject (string): subject id

    Returns:
        string: the path string for this subject
    """
    return "raw_data/" + subject + "_3T_Diffusion_preproc/" + subject + "/T1w/Diffusion/"

def restore_duplicates(tensor):
    """Creates a flattened diffusion tensor from 6 unique diffusion parameters

    Args:
        tensor ([double]): array containing 6 diffusion parameters

    Returns:
        [double]: a diffusion tensor with 9 elements
    """
    d_yx = tensor[1]
    d_zx = tensor[2]
    d_zy = tensor[4]
    new_tensor = np.array([[tensor[0], tensor[1], tensor[2]],
                  [d_yx, tensor[3], tensor[4]],
                  [d_zx, d_zy, tensor[5]]])
    
    return new_tensor

def create_triples(x_max, y_max, z_max):
    """Generate all possible coordinates in a given 3D space starting from 0

    Args:
        x_max (int): maximum x coordinate
        y_max (int): maximum y coordinate
        z_max (int): maximum z coordinate

    Returns:
        [(int, int, int)]: list of triples
    """
    triples = []
    for x in range(0, x_max):
        for y in range(0, y_max):
            for z in range(0, z_max):
                triples.append((x, y, z))
    return triples

def complete_patch(p_mask, p_patch, mean, covariance):
    # calculate the conditional mean given a subset of components
    p_patch = p_patch.flatten()
    p_mask = np.repeat(p_mask[:, :, :, np.newaxis], 6, axis=3)
    p_mask = p_mask.flatten()
    missing_idx = p_mask == False
    known_idx = p_mask == True
    
    u1 = mean[missing_idx]
    u2 = mean[known_idx]
    cov12 = covariance[missing_idx, :][:, known_idx]
    cov22_inv = np.linalg.inv(covariance[known_idx, :][:, known_idx])
    diff = p_patch[known_idx] - u2
    
    cond_mean = u1 + np.linalg.multi_dot([cov12, cov22_inv, diff])
    
    # insert the conditional mean
    p_patch[missing_idx] = cond_mean
    return p_patch