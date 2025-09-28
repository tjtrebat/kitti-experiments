import numpy as np


def inverse_rigid_transform(transform):
    inv = np.zeros_like(transform)
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    inv[0:3, 0:3] = R.T
    inv[0:3, 3] = -R.T @ t
    inv[3, 3] = 1.0
    return inv
