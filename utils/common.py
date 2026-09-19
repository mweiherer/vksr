import numpy as np
import torch
import trimesh
import point_cloud_utils as pcu


def offset_along_normal(points, normals, surf_eps):
    """
    Offset surface points along positive and negative normal direction by a small epsilon.
    :param points: A batch of surface points as torch.Tensor of size [b, n, 3]
    :param normals: Corresponding surface normals as torch.Tensor of size [b, n, 3]
    :param surf_eps: Small epsilon by which to offset the points
    :return: offset_points as torch.Tensor of size [b, 2n, 3] and 
             offset_occs as torch.Tensor of size [b, 2n, 1]
    """
    offset_points = torch.cat([points + normals * surf_eps,
                               points - normals * surf_eps], dim = 1).to(points)
    offset_occs = torch.cat([torch.ones(points.shape[0], points.shape[1]) * surf_eps,
                             -torch.ones(points.shape[0], points.shape[1]) * surf_eps], dim = 1).to(points)
    return offset_points, offset_occs.unsqueeze(dim = -1)

# Slightly adapted from: https://github.com/fwilliams/neural-splines/blob/main/trim-surface.py.
def trim_reconstructed_mesh(mesh, input_points, trim_distance):
    """
    Trims a reconstructed surface mesh.
    :param mesh: The reconstructed surface mesh as a trimesh object
    :param input_points: The original input point cloud as a numpy array of size [n, 3]
    :param trim_distance: Distance threshold (in voxels) used to trim away spurious geometry, if any
    :return: The trimmed surface mesh as a trimesh object
    """
    vertices, faces = mesh.vertices, mesh.faces
    nn_dist, _ = pcu.k_nearest_neighbors(vertices, input_points, k = 2)
    nn_dist = nn_dist[:, 1]
    f_mask = np.stack([nn_dist[faces[:, i]] < trim_distance for i in range(faces.shape[1])], axis = -1)
    f_mask = np.all(f_mask, axis = -1)
    faces = faces[f_mask]
    return trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = mesh.vertex_normals, process = False)

# Slightly adapted from: https://github.com/fwilliams/neural-splines/blob/main/neural_splines/geometry.py. 
def point_cloud_bounding_box(x, uniform = False, scale = 1.0):
    """
    Get the axis-aligned bounding box for a point cloud (possibly scaled by some factor)
    :param x: A point cloud represented as an [N, 3]-shaped tensor
    :param uniform: Whether to use a uniform (regular) bounding box (i.e., all sides equal length)
    :param scale: A scale factor by which to scale the bounding box diagonal
    :return: The (possibly scaled) axis-aligned bounding box for a point cloud represented as a pair (origin, size)
    """
    bb_min = x.min(0)[0]
    bb_size = x.max(0)[0] - bb_min
    if uniform: 
        bb_size = bb_size.max()
    return scale_bounding_box_diameter((bb_min, bb_size), scale)

# Source: https://github.com/fwilliams/neural-splines/blob/main/neural_splines/geometry.py.
def scale_bounding_box_diameter(bbox, scale):
    """
    Scale the diagonal of the bounding box while maintaining its center position
    :param bbox: A bounding box represented as a pair (origin, size)
    :param scale: A scale factor by which to scale the input bounding box's diagonal
    :return: The (possibly scaled) axis-aligned bounding box for a point cloud represented as a pair (origin, size)
    """
    bb_min, bb_size = bbox
    bb_diameter = torch.norm(bb_size)
    bb_unit_dir = bb_size / bb_diameter
    scaled_bb_size = bb_size * scale
    scaled_bb_diameter = torch.norm(scaled_bb_size)
    scaled_bb_min = bb_min - 0.5 * (scaled_bb_diameter - bb_diameter) * bb_unit_dir
    return scaled_bb_min, scaled_bb_size