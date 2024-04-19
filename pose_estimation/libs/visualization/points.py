"""
Simple visualization utilities for 2D and 3D points based on Matplotlib.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def check_points(points, dimension):
    """
    Assertion function for input dimension.
    """
    if len(points.shape) == 1:
        assert points.shape[0] % dimension == 0
        points = points.reshape(-1, dimension)
    elif len(points.shape) == 2:
        assert points.shape[1] == dimension
    else:
        raise ValueError    
    return points

def set_3d_axe_limits(ax, points=None, center=None, radius=None, ratio=1.2):
    """
    Set 3d axe limits to simulate set_aspect('equal').
    Matplotlib has not yet provided implementation of set_aspect('equal') 
    for 3d axe.
    """
    if points is None:
        assert center is not None and radius is not None
    if center is None or radius is None:
        assert points is not None
    if center is None:
        center = points.mean(axis=0, keepdims=True)
    if radius is None:
        radius = points - center
        radius = np.max(np.abs(radius))*ratio
    #ax.set_aspect('equal')    
    xroot, yroot, zroot = center[0,0], center[0,1], center[0,2]
    ax.set_xlim3d([-radius+xroot, radius+xroot])
    ax.set_ylim3d([-radius+yroot, radius+yroot])
    ax.set_zlim3d([-radius+zroot, radius+zroot])    
    return

def plot_3d_points(ax, 
                   points, 
                   indices=None, 
                   center=None, 
                   radius=None,  
                   add_labels=True, 
                   display_ticks=True, 
                   remove_planes=[],
                   marker='o', 
                   color='k', 
                   size=50, 
                   alpha=1, 
                   set_limits=False
                   ):
    """
    Scatter plot of 3D points.
    
    points are of shape [3*N_points] or [N_points, 3]
    """
    points = check_points(points, dimension=3)
    points = points[indices,:] if indices is not None else points
    ax.scatter(points[:,0], points[:,1], points[:,2], marker=marker, c=color,
               s=size, alpha=alpha)
    if set_limits:
        set_3d_axe_limits(ax, points, center, radius)
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    # remove tick labels or planes
    if not display_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])
    white = (1.0, 1.0, 1.0, 1.0)
    if 'x' in remove_planes:
        ax.w_xaxis.set_pane_color(white)
    if 'y' in remove_planes:
        ax.w_xaxis.set_pane_color(white)
    if 'z' in remove_planes:
        ax.w_xaxis.set_pane_color(white)        
    
    plt.show()
    return

def plot_lines(ax, 
               points, 
               connections, 
               dimension, 
               lw=4, 
               c='k', 
               linestyle='-', 
               alpha=0.8, 
               add_index=False
               ):
    """
    Plot 2D/3D lines given points and connection.
    
    connections are of shape [n_lines, 2]
    """
    points = check_points(points, dimension)
    if add_index:
        for idx in range(len(points)):
            if dimension == 2:
                x, y = points[idx][0], points[idx][1]
                ax.text(x, y, str(idx))
            elif dimension == 3:
                x, y, z = points[idx][0], points[idx][1], points[idx][2]
                ax.text(x, y, z, str(idx))                
    connections = connections.reshape(-1, 2)
    for connection in connections:
        x = [points[connection[0]][0], points[connection[1]][0]]
        y = [points[connection[0]][1], points[connection[1]][1]]
        if dimension == 3:
            z = [points[connection[0]][2], points[connection[1]][2]]
            line, = ax.plot(x, y, z, lw=lw, c=c, linestyle=linestyle, alpha=alpha)
        else:
            line, = ax.plot(x, y, lw=lw, c=c, linestyle=linestyle, alpha=alpha)
    plt.show()
    return line

def plot_mesh(ax, vertices, faces, color='grey'):
    """
    Simple mesh plotting.
    
    vertics of shape [N_vertices, 3]
    faces pf shape [N_faces, 3] storing indices
    """    
    set_3d_axe_limits(ax, vertices)
    ax.plot_trisurf(vertices[:, 0], 
                    vertices[:, 1], 
                    faces, 
                    -vertices[:, 2], 
                    shade=True, 
                    color=color
                    )    
    return

def plot_3d_coordinate_system(ax, 
                              origin, 
                              system, 
                              length=300, 
                              colors=['r', 'g', 'b']
                              ):
    """
    Draw a coordinate system at a specified origin
    
    system: [v1, v2, v3] 
    """ 
    origin = origin.reshape(3, 1)
    start_points = np.repeat(origin, 3, axis=1)
    end_points = start_points + system*length
    all_points = np.hstack([origin, end_points])
    for i in range(3):
        plot_lines(ax, 
                   all_points.T, 
                   plot_3d_coordinate_system.connections[i].reshape(1,2),
                   dimension=3, 
                   c=colors[i]
                   )
    return

def plot_3d_bbox(ax, 
                 bbox_3d_projected, 
                 color=None, 
                 linestyle='-', 
                 add_index=False
                 ):
    """
    Draw the projected edges of a 3D cuboid.
    """ 
    c = np.random.rand(3) if color is None else color
    plot_lines(ax, 
               bbox_3d_projected, 
               plot_3d_bbox.connections, 
               dimension=2, 
               c=c, 
               linestyle=linestyle, 
               add_index=add_index
               )
    return

def plot_2d_bbox(ax, 
                 bbox_2d, 
                 color=None, 
                 score=None, 
                 label=None, 
                 linestyle='-'
                 ):
    """
    Draw a 2D bounding box.
    
    bbox_2d in the format [x1, y1, x2, y2]
    """ 
    c = np.random.rand(3) if color is None else color
    x1, y1, x2, y2 = bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    plot_lines(ax, points, plot_2d_bbox.connections, dimension=2, c=c, linestyle=linestyle)
    if score is not None and label is not None:
        string = "({:.2f}, {:d})".format(score, label)
        ax.text((x1+x2)*0.5, (y1+y2)*0.5, string, bbox=dict(facecolor='red', alpha=0.2))
    return

def plot_comparison_relative(points_pred, points_gt):
    # DEPRECATED
    # plot the comparison of the shape relative to the root point
    plt.figure()
    num_row = 3
    num_col = int(len(points_pred)/num_row)
    for i in range(len(points_pred)):
        ax = plt.subplot(num_row, num_col, i+1, projection='3d')
        pred = points_pred[i]
        gt = points_gt[i]
        plot_3d_points(ax, pred, color='r')
        plot_3d_points(ax, gt, color='k')
        pred_bbox = get_bbox_3d(pred)
        gt_bbox = get_bbox_3d(gt)
        plot_lines(ax, pred_bbox, plot_3d_bbox.connections, dimension=3, c='r')
        plot_lines(ax, gt_bbox, plot_3d_bbox.connections, dimension=3, c='k')
        set_3d_axe_limits(ax, 
                          np.vstack([pred_bbox.reshape(-1, 3), 
                                     gt_bbox.reshape(-1, 3)]
                                    ),
                          center=np.zeros((1,3)), 
                          radius=5.
                          )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")    
    ax.view_init(0., 90.)
    return

def plot_scene_3dbox(points_pred, points_gt=None, ax=None, color='r'):
    """
    Plot the comparison of predicted 3d bounding boxes and ground truth ones.
    """ 
    if ax is None:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
    preds = points_pred.copy()
    # add the root translation
    preds[:,1:,] = preds[:,1:,] + preds[:,[0],]
    if points_gt is not None:
        gts = points_gt.copy() 
        gts[:,1:,] = gts[:,1:,] + gts[:,[0],]
        all_points = np.concatenate([preds, gts], axis=0).reshape(-1, 3)
    else:
        all_points = preds.reshape(-1, 3)
    for pred in preds:
        plot_3d_points(ax, pred, color=color, size=15)
        plot_lines(ax, pred[1:,], plot_3d_bbox.connections, dimension=3, c=color)
    if points_gt is not None:
        for gt in gts:
            plot_3d_points(ax, gt, color='k', size=15)
            plot_lines(ax, gt[1:,], plot_3d_bbox.connections, dimension=3, c='k')         
    set_3d_axe_limits(ax, all_points)
    return ax

def get_area(points, indices, preserve_points=False):
    # DEPRECATED
    # points [N, 2]
    # indices [M, 3]
    vec1 = points[indices[:, 1], :] - points[indices[:, 0], :]
    vec2 = points[indices[:, 2], :] - points[indices[:, 0], :]
    area= np.cross(vec1, vec2)*0.5
    area = area.reshape(1, -1)
    if preserve_points:
        feature = np.hstack([points.reshape(1,-1), area])
    else:
        feature = area
    return feature

def interpolate(start, end, num_interp):
    # DEPRECATED
    # start: [3]
    # end: [3]
    x = np.linspace(start[0], end[0], num=num_interp+2)[1:-1].reshape(num_interp, 1)
    y = np.linspace(start[1], end[1], num=num_interp+2)[1:-1].reshape(num_interp, 1)
    z = np.linspace(start[2], end[2], num=num_interp+2)[1:-1].reshape(num_interp, 1)
    return np.concatenate([x,y,z], axis=1)

def get_interpolated_points(points, indices, num_interp):
    # DEPRECATED
    # points [N, 3]
    # indices [M, 2] point indices for interpolating a line segment
    # num_interp how many points to add for each segment
    new_points = []
    for start_idx, end_idx in indices:
        new_points.append(interpolate(points[start_idx], points[end_idx], num_interp))
    return np.vstack(new_points)

def draw_pose_vecs(ax, pose_vecs=None, color='black'):
    """
    Add pose vectors to a 3D matplotlib axe.
    """     
    if pose_vecs is None:
        return
    for pose_vec in pose_vecs:
        x, y, z, pitch, yaw, roll = pose_vec
        string = "({:.2f}, {:.2f}, {:.2f})".format(pitch, yaw, roll)
        # add some random noise to the text location so that they do not overlap
        nl = 0.02 # noise level
        ax.text(x*(1+np.random.randn()*nl), 
                y*(1+np.random.randn()*nl), 
                z*(1+np.random.randn()*nl), 
                string, 
                color=color
                )
        
def get_bbox_3d(points, add_center=False, interp_style=""):
    """
    Get a 3D bounding boxes from coordinate limits in object coordinate system.
    """  
    assert len(points.shape) == 2 
    if points.shape[0] == 3:
        axis=1 
    elif points.shape[1] == 3:
        axis=0
    limit_min = points.min(axis=axis)
    limit_max = points.max(axis=axis)
    xmax, xmin = limit_max[0], limit_min[0]
    ymax, ymin = limit_max[1], limit_min[1]
    zmax, zmin = limit_max[2], limit_min[2]
    bbox = np.array([[xmax, ymin, zmax],
                     [xmax, ymax, zmax],
                     [xmax, ymin, zmin],
                     [xmax, ymax, zmin],
                     [xmin, ymin, zmax],
                     [xmin, ymax, zmax],
                     [xmin, ymin, zmin],
                     [xmin, ymax, zmin]])
    if add_center:
        bbox = np.vstack([np.array([[0., 0., 0.]]), bbox])
    if interp_style.startswith('bbox9interp'):
        interp_num = int(interp_style[11:])
        # indices for each edge
        indices = np.array([[1,2],
                            [3,4],
                            [1,3],
                            [2,4],
                            [5,6],
                            [7,8],
                            [5,7],
                            [6,8],
                            [1,5],
                            [3,7],
                            [2,6],
                            [4,8]])
        new_points = get_interpolated_points(bbox, indices, interp_num)
        bbox = np.vstack([bbox, new_points])
    return bbox

def ray_intersect_triangle(p0, p1, triangle):
    """
    Tests if a ray starting at point p0, in the direction
    p1 - p0, will intersect with the triangle.
    
    arguments:
    p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    triangle: numpy.ndarray, shaped (3,3), with each row
        representing a vertex and three columns for x, y, z.
    
    returns: 
        0.0 if ray does not intersect triangle, 
        1.0 if it will intersect the triangle,
        2.0 if starting point lies in the triangle.
        
    Reference: https://www.erikrotteveel.com/python/three-dimensional-ray-tracing-in-python/
    """
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)
    if (b == 0.0):
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b
    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom
    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom
    if (ti < 0.0) | (si + ti > 1.0):
        return 0
    if (rI == 0.0):
        return 2
    return 1

def get_visibility(box3d, triangles):
    """
    Get visibility for each vertex of a 3D bounding box given all the triangles
    in a scene.
    
    box3d: [8, 3] The vertex coordinates in the camera coordinate system.
    triangles: [N, 3, 3]
    """      
    visibility = np.ones(8, dtype=np.bool)
    p1 = np.zeros(3)
    for idx, p0 in enumerate(box3d):
        intersects = set()
        for triangle in triangles:
            intersects.add(ray_intersect_triangle(p0, p1, triangle))
        if 1 in intersects:
            visibility[idx] = False
    return visibility

## static variables implemented as function attributes 
plot_3d_coordinate_system.connections = np.array([[0, 1],
                                                  [0, 2],
                                                  [0, 3]])
plot_3d_bbox.connections = np.array([[0, 1],
                                     [0, 2],
                                     [1, 3],
                                     [2, 3],
                                     [4, 5],
                                     [5, 7],
                                     [4, 6],
                                     [6, 7],
                                     [0, 4],
                                     [1, 5],
                                     [2, 6],
                                     [3, 7]])
plot_2d_bbox.connections = np.array([[0, 1],
                                     [1, 2],
                                     [2, 3],
                                     [3, 0]])