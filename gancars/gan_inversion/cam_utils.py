import numpy as np
import cv2 as cv
from collections import namedtuple


CoordSystem = namedtuple("CoordSystem", ["basis", "origin"])

eg3d_object_sys = CoordSystem(
    basis=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    origin=np.array([[0, 0, 0]]),
)

def get_transform_in_sys(object_sys, transformation):
    to_world = np.eye(4)
    to_world[:3, -1] = -object_sys.origin
    to_sys_center = np.eye(4)
    to_sys_center[:3, -1] = object_sys.origin
    algin_axes_to_world = np.eye(4)
    algin_axes_to_world[:3, :3] = object_sys.basis
    return to_sys_center @ np.linalg.inv(algin_axes_to_world) @ transformation @ algin_axes_to_world @ to_world


def transform_coord_system(coord_system: CoordSystem, transformation):
    new_origin = transformation @ np.block([coord_system.origin, 1.0]).T
    new_origin /= new_origin[-1]
    new_origin = new_origin[:3].T

    new_basis = (transformation @ np.block([coord_system.basis, np.ones((coord_system.basis.shape[0], 1))]).T).T
    new_basis = new_basis[:, :3] - new_origin
    return CoordSystem(new_basis, new_origin)


def get_rotation_mat_from_cam(cam):

    rvec = np.array([0, 0, -np.pi/2])
    M = np.eye(4)
    M[:3,:3] = cv.Rodrigues(rvec)[0]
    M = get_transform_in_sys(cam, M)

    rvec = np.array([0, np.pi, 0])
    M2 = np.eye(4)
    M2[:3,:3] = cv.Rodrigues(rvec)[0]
    M2 = get_transform_in_sys(cam, M2)

    return M, M2


def blender2eg3d(cam_mat):

    rvec = np.array([-np.pi/2, 0, 0])
    M = np.eye(4)
    M[:3,:3] = cv.Rodrigues(rvec)[0]
    M_world_rot = get_transform_in_sys(eg3d_object_sys, M)

    M, M2 = get_rotation_mat_from_cam(transform_coord_system(eg3d_object_sys, cam_mat)) 
            
    cam_mat_eg3d = M_world_rot @ M2 @ M @ cam_mat

    # cam_mat_ingp = M_world_rot @ np.linalg.inv(M2 @ M) @ cam_mat
    # cars were upside down
    rvec = np.array([0, 0, -np.pi/2.0])
    M3 = np.eye(4)
    M3[:3,:3] = cv.Rodrigues(rvec)[0]
    M3 = get_transform_in_sys(transform_coord_system(eg3d_object_sys, cam_mat_eg3d), M3)
    
    return M3 @ cam_mat_eg3d