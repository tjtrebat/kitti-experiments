import numpy as np


def inverse_rigid_transform(transform):
    inv = np.zeros_like(transform)
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    inv[0:3, 0:3] = R.T
    inv[0:3, 3] = -R.T @ t
    inv[3, 3] = 1.0
    return inv


def transform_points(points, transform):
    points_hom = np.c_[points, np.ones(points.shape[0])]
    transformed = points_hom @ transform.T
    return transformed[:, :3]


def compute_box_3d(label, transform):
    h, w, l = label["dimensions"]
    x, y, z = label["centroid"]
    ry = label["rotation_y"]
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [    0,    0,    0,    0,  -h,  -h,  -h,  -h]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners + np.array([[x], [y], [z]])
    return transform_points(corners_3d.T, transform)
