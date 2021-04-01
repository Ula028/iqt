
import numpy as np

def join_path(subject):
    """Returns the path for diffusion data for a particular subject.

    Args:
        subject (string): subject id

    Returns:
        string: the path string for this subject
    """
    return "raw_data/" + subject + "_3T_Diffusion_preproc/" + subject + "/T1w/Diffusion/"


def contained_in_brain(index, radius, mask):
    """Return True if the patch is contained within the brain or False otherwise.
    
    Checks if every corner of the cubic patch is contained within the brain mask, and if yes,
    assumes that the whole patch is contained within the brain mask.

    Args:
        index (int, int, int): central index of the patch
        radius (int): radius of the patch
        mask ([type]): brain mask

    Returns:
        boolean: True if the cubic patch is contained within the brain, False otherwise
    """
    n = radius
    x, y, z = index
    if mask[x + n, y + n, z + n] and \
            mask[x + n, y + n, z - n] and \
            mask[x + n, y - n, z + n] and \
            mask[x + n, y - n, z - n] and \
            mask[x - n, y + n, z + n] and \
            mask[x - n, y + n, z - n] and \
            mask[x - n, y - n, z + n] and \
            mask[x - n, y - n, z - n]:
        return True
    else:
        return False

def restore_duplicates(tensor):
    print(tensor.shape)
    d_yx = tensor[1]
    d_zx = tensor[2]
    d_zy = tensor[4]
    tensor = np.insert(tensor, 5, d_yx)
    tensor = np.insert(tensor, 5, d_zx)
    tensor = np.insert(tensor, 3, d_zy)
    
    print(tensor.shape)
    return tensor

def create_triples(x_max, y_max, z_max):
    triples = []
    for x in range(0, x_max):
        for y in range(0, y_max):
            for z in range(0, z_max):
                triples.append((x, y, z))
    return triples