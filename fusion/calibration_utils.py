import numpy as np


def convert_to_img_coords(points_velo, calib):
    points_cam = transform_to_cam_coords(points_velo, calib)
    points_cam_hom = np.vstack((points_cam, np.ones((1, points_cam.shape[1]))))
    u, v = project_to_image_plane(points_cam_hom, calib['P2'])
    return np.stack((u, v), axis=1)

   
def transform_to_cam_coords(points_velo, calib):
    R0 = calib['R0']
    Tr = calib['Tr_velo_to_cam']
    points_hom = np.hstack((points_velo[:, :3], np.ones((points_velo.shape[0], 1))))
    points_cam = (R0 @ (Tr @ points_hom.T)[:3, :])
    return points_cam


def project_to_image_plane(points_cam_hom, P2):
    projections = P2 @ points_cam_hom
    u = projections[0, :] / projections[2, :]
    v = projections[1, :] / projections[2, :]
    return u, v
