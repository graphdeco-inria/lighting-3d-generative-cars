import json
import cv2 as cv
import numpy as np
import polyscope as ps
from collections import namedtuple

ObjectBox = namedtuple("ObjectBox", ["coord_system", "vertices", "faces", "name"])
CoordSystem = namedtuple("CoordSystem", ["basis", "origin"])

interp_coef = [0.332, 0.667]
faces = np.array(
    [
        [3, 2, 0],
        [3, 1, 0],
        [3, 2, 6],
        [3, 7, 6],
        [1, 0, 4],
        [1, 5, 4],
        [7, 5, 4],
        [7, 6, 4],
        [7, 5, 1],
        [7, 3, 1],
        [6, 2, 0],
        [6, 4, 0],
    ]
)

interp_dict = {
    "bbox12": (
        np.array(
            [1, 3, 5, 7, 1, 2, 3, 4, 1, 2, 5, 6]  # h direction  # l direction
        ),  # w direction
        np.array([2, 4, 6, 8, 5, 6, 7, 8, 3, 4, 7, 8]),
    ),
    "bbox12l": (
        np.array(
            [
                1,
                2,
                3,
                4,
            ]
        ),  # w direction
        np.array([5, 6, 7, 8]),
    ),
    "bbox12h": (np.array([1, 3, 5, 7]), np.array([2, 4, 6, 8])),  # w direction
    "bbox12w": (np.array([1, 2, 5, 6]), np.array([3, 4, 7, 8])),  # w direction
}


def get_box_height(prediction):
    parents = prediction[interp_dict["bbox12"][0] - 1]
    children = prediction[interp_dict["bbox12"][1] - 1]
    lines = parents - children
    lines = np.sqrt(np.sum(lines**2, axis=1))
    h = np.sum(lines[:4]) / 4
    return h


def get_transform_in_sys(object_sys, transformation):
    to_world = np.eye(4)
    to_world[:3, -1] = -object_sys.origin
    to_sys_center = np.eye(4)
    to_sys_center[:3, -1] = object_sys.origin
    algin_axes_to_world = np.eye(4)
    algin_axes_to_world[:3, :3] = object_sys.basis
    return (
        to_sys_center
        @ np.linalg.inv(algin_axes_to_world)
        @ transformation
        @ algin_axes_to_world
        @ to_world
    )


def transform_coord_system(coord_system: CoordSystem, transformation):
    new_origin = transformation @ np.block([coord_system.origin, 1.0]).T
    new_origin /= new_origin[-1]
    new_origin = new_origin[:3].T

    new_basis = (
        transformation
        @ np.block([coord_system.basis, np.ones((coord_system.basis.shape[0], 1))]).T
    ).T
    new_basis = new_basis[:, :3] - new_origin
    return CoordSystem(new_basis, new_origin)


def create_canonical_box(l=2, h=1, w=1, prediction=None):
    if prediction is not None:
        parents = prediction[interp_dict["bbox12"][0] - 1]
        children = prediction[interp_dict["bbox12"][1] - 1]
        lines = parents - children
        lines = np.sqrt(np.sum(lines**2, axis=1))
        # averaged over the four parallel line segments
        h, l, w = (
            np.sum(lines[:4]) / 4,
            np.sum(lines[4:8]) / 4,
            np.sum(lines[8:]) / 4,
        )

    x_corners = [w, w, 0, 0, w, w, 0, 0]
    y_corners = [0, h, 0, h, 0, h, 0, h]
    z_corners = [l, l, l, l, 0, 0, 0, 0]
    x_corners -= np.float32(w) / 2.0
    y_corners -= np.float32(h) / 2.0
    z_corners -= np.float32(l) / 2.0
    corners_3d = np.array([x_corners, y_corners, z_corners])
    return corners_3d.T


def create_box(transformation, coord_system, vertices, faces, name):
    new_vertices = (
        transformation @ np.block([vertices, np.ones((vertices.shape[0], 1))]).T
    ).T
    new_vertices = new_vertices[:, :3]
    new_coord_system = transform_coord_system(coord_system, transformation)
    return ObjectBox(new_coord_system, new_vertices, faces, name)


def add_box(box: ObjectBox, color):
    ps.register_surface_mesh(
        box.name, box.vertices, box.faces, color=color, transparency=0.6
    )
    add_coords(box.name, box.coord_system)


def add_coords(name: str, coord_system: CoordSystem):
    coord_sys = ps.register_point_cloud(
        f"{name} sys", coord_system.origin, radius=0.02, color=(1, 1, 1)
    )
    coord_sys.add_vector_quantity(
        "X",
        coord_system.basis[0][None, :],
        length=0.1,
        enabled=True,
        radius=0.01,
        color=(1, 0, 0),
    )
    coord_sys.add_vector_quantity(
        "Y",
        coord_system.basis[1][None, :],
        length=0.1,
        enabled=True,
        radius=0.01,
        color=(0, 1, 0),
    )
    coord_sys.add_vector_quantity(
        "Z",
        coord_system.basis[2][None, :],
        length=0.1,
        enabled=True,
        radius=0.01,
        color=(0, 0, 1),
    )


def transform_points(object_sys, points, transformation):
    to_world = np.eye(4)
    to_world[:3, -1] = -object_sys.origin
    to_sys_center = np.eye(4)
    to_sys_center[:3, -1] = object_sys.origin
    algin_axes_to_world = np.eye(4)
    algin_axes_to_world[:3, :3] = object_sys.basis
    M = (
        to_sys_center
        @ np.linalg.inv(algin_axes_to_world)
        @ transformation
        @ algin_axes_to_world
        @ to_world
    )
    new_points = (M @ np.block([points, np.ones((points.shape[0], 1))]).T).T
    new_points = new_points[:, :3]
    return new_points


def generate_eg3d_cam2world(points_2d, points_3d, intrinsics):

    world2cam = np.eye(4)
    _, rvec, tvec = cv.solvePnP(points_3d, points_2d, intrinsics, None)
    rvec[1] *= -1
    world2cam[:3, :3] = cv.Rodrigues(rvec)[0]
    world2cam[:3, -1] = tvec[:, 0]
    cam2world = np.linalg.inv(world2cam)
    eg3d_camera = transform_coord_system(eg3d_object_sys, cam2world)

    align_axes = np.eye(4, dtype=np.float32)
    align_axes[2, 2] = -1
    align_axes[0, 0] = -1
    align_axes = get_transform_in_sys(eg3d_camera, align_axes)

    cam2world = align_axes @ cam2world
    return cam2world


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
    return (
        to_sys_center
        @ np.linalg.inv(algin_axes_to_world)
        @ transformation
        @ algin_axes_to_world
        @ to_world
    )


def transform_coord_system(coord_system: CoordSystem, transformation):
    new_origin = transformation @ np.block([coord_system.origin, 1.0]).T
    new_origin /= new_origin[-1]
    new_origin = new_origin[:3].T

    new_basis = (
        transformation
        @ np.block([coord_system.basis, np.ones((coord_system.basis.shape[0], 1))]).T
    ).T
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

def ingp2eg3d(cam_mat):

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

def campari2eg3d(cam_mat):
    M = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1.0]
    ])
    M_world_rot = get_transform_in_sys(eg3d_object_sys, M)
    M, M2 = get_rotation_mat_from_cam(transform_coord_system(eg3d_object_sys, cam_mat))  
    cam_mat_eg3d = M_world_rot @ M2 @ M @ cam_mat

    rvec = np.array([0, 0, -np.pi/2.0])
    M3 = np.eye(4)
    M3[:3,:3] = cv.Rodrigues(rvec)[0]
    M3 = get_transform_in_sys(transform_coord_system(eg3d_object_sys, cam_mat_eg3d), M3)

    return M3 @ cam_mat_eg3d


if __name__ == "__main__":
    with open("./camera_calib/data/dataset.json", "r+") as f:
        data = json.load(f)

    # Load annoations
    cams = {}
    for annot in data["labels"]:
        num = annot[0][3:-4]
        cam2world = np.array(annot[1][:16]).reshape(4, 4)
        cam = transform_coord_system(eg3d_object_sys, cam2world)
        cams[num] = cam

    canonical_box = ObjectBox(
        eg3d_object_sys, create_canonical_box(3 / 5, 1.5 / 5, 1.5 / 5), faces, "box"
    )

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    add_coords("EG3D", eg3d_object_sys)
    for name, cam in cams.items():
        add_coords(f"cam {name}", cam)
    add_box(canonical_box, (0, 1, 0))
    ps.show()
