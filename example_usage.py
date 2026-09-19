import torch
import point_cloud_utils as pcu

from vksr import Reconstructor


if __name__ == '__main__':
    device = 'cuda'

    input_points, input_normals = pcu.load_mesh_vn('./demo_data/bunny.ply')

    input_points = torch.from_numpy(input_points)
    input_normals = torch.from_numpy(input_normals)

    reconstructor = Reconstructor(device = device)

    mesh = reconstructor.reconstruct(input_points, input_normals, k = 32, reg_weight = 1e-5)
    mesh.export('reconstruction.ply')