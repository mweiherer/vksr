import torch
import trimesh
from skimage.measure import marching_cubes

from vksr.kernel import MaternKernel
from vksr.knn_searcher import ExactFaissNearestNeighborSearcher, ApproximateFaissNearestNeighborSearcher
from vksr.solver import CholeskyKernelSolver
from utils.common import offset_along_normal, point_cloud_bounding_box, trim_reconstructed_mesh
from utils.mise_gpu import MISE


knn_searcher_dict = {
    'exact_faiss': ExactFaissNearestNeighborSearcher,
    'approximate_faiss': ApproximateFaissNearestNeighborSearcher
}

kernel_dict = {
    'matern': MaternKernel
}

solver_dict = {
    'cholesky': CholeskyKernelSolver
}


class Reconstructor:
    def __init__(self, 
                 knn_searcher = 'approximate_faiss',
                 knn_searcher_kwargs = {'nlist': 1000, 'nprobe': 10},
                 kernel = 'matern', 
                 kernel_kwargs = {'order': '1/2', 'h': 1.0},
                 solver = 'cholesky',
                 solver_kwargs = {},
                 use_abs_units = False,
                 device = 'cuda'):
        
        self.knn_searcher = knn_searcher_dict[knn_searcher](device, **knn_searcher_kwargs)
        self.solver = solver_dict[solver](kernel_dict[kernel](**kernel_kwargs), **solver_kwargs)   

        self.use_abs_units = use_abs_units
        self.device = device 

    def _local_regression(self, query_points, input_points, input_normals, k, reg_weight, chunk_size, surf_eps):
        chunks = torch.split(query_points, chunk_size)

        fx = []
        for chunk in chunks:
            I = self.knn_searcher.query(chunk.cpu(), k)
            knn_points, knn_normals = input_points[I], input_normals[I]
            offset_points, offset_occs = offset_along_normal(knn_points, knn_normals, surf_eps)
            fx_i = self.solver.fit_and_predict(offset_points.to(self.device), offset_occs.to(self.device),
                                               reg_weight, chunk.unsqueeze(dim = 1).to(self.device))
            fx.append(fx_i.squeeze(dim = 1))

        return torch.cat(fx, dim = 0).squeeze(dim = -1)

    def reconstruct(self, input_points, input_normals, k, reg_weight, chunk_size = 1000, surf_eps = 0.05,
                    voxel_resolution_0 = 32, num_upsampling_steps = 3, trim = 0.0):
        """
        Reconstructs a surface mesh from an oriented point cloud.
        :param input_points: Input point cloud as torch.Tensor of size [n, 3]
        :param input_normals: Corresponding surface normals as torch.Tensor of size [n, 3]
        :param k: Number of nearest neighbors
        :param reg_weight: Regularization weight
        :param chunk_size: Number of query points to process in each chunk
        :param surf_eps: Offset distance along the normals
        :param voxel_resolution_0: Initial voxel resolution for MISE algorithm
        :param num_upsampling_steps: Number of upsampling steps for MISE algorithm
        :param trim: Distance threshold (in voxels) used to trim away spurious geometry, if any
        :return: Reconstructed surface mesh as a trimesh object
        """
        torch.cuda.reset_peak_memory_stats() 
      
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()

        # Normalize input point cloud to [-0.5, 0.5]^3.
        bbox_min, bbox_max = input_points.min(0)[0], input_points.max(0)[0]
        bb_center = (bbox_min + bbox_max) / 2.0
        scale = 1.0 / (bbox_max - bbox_min).max()
        input_points = (input_points - bb_center) * scale

        bb_min, bb_size = point_cloud_bounding_box(input_points, uniform = True, scale = 1.1)
        voxel_size = bb_size / (voxel_resolution_0 * (2 ** num_upsampling_steps))
        voxel_diag = (3 ** 0.5) * voxel_size

        if not self.use_abs_units:
            surf_eps = surf_eps * voxel_diag

        # Build NN search index.
        self.knn_searcher.build_index(input_points)
 
        mesh_extractor = MISE(voxel_resolution_0, num_upsampling_steps, threshold = 0, device = self.device)
        points_0 = mesh_extractor.query()
        
        # Prune away points that are outside the input's bounding box.
        non_uni_bb_size = input_points.max(0)[0] - input_points.min(0)[0]
        non_uni_grid_resolution_0 = torch.round(non_uni_bb_size / non_uni_bb_size.max() * voxel_resolution_0).to(torch.int32).numpy()
        mask_0 = (points_0[:, 0] < non_uni_grid_resolution_0[0]) & (points_0[:, 1] < non_uni_grid_resolution_0[1]) & \
            (points_0[:, 2] < non_uni_grid_resolution_0[2]) 

        points = points_0[mask_0]

        while points.shape[0] != 0:
            query_points = torch.from_numpy((points / mesh_extractor.resolution)).to(input_points) 
            query_points = (query_points * bb_size) + bb_min

            fx = self._local_regression(query_points, input_points, input_normals, k, reg_weight, chunk_size, surf_eps)

            if mesh_extractor.resolution == voxel_resolution_0:
                fx_0 = torch.zeros(mask_0.shape[0])
                fx_0[mask_0], fx_0[~mask_0] = fx, 1e10 # Set values for points outside the input's bounding box to a large number.
                mesh_extractor.update(points_0, fx_0)
            else:
                mesh_extractor.update(points, fx) 

            points = mesh_extractor.query()
            
        # Extract surface mesh using standard Marching Cubes on the CPU.
        dense = mesh_extractor.to_dense()
        vertices, faces, normals, _ = marching_cubes(dense, level = 0, spacing = [voxel_size] * 3)
        vertices += bb_min.numpy()

        mesh = trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = normals, process = False)

        end.record()
        torch.cuda.synchronize()

        total_runtime = start.elapsed_time(end) / 1000.0
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'Reconstruction finished. Total runtime: {total_runtime:.2f} s, peak GPU memory: {peak_gpu_memory:.2f} MB.')

        if trim > 0.0:
            print('Trimming extracted mesh...')

            if not self.use_abs_units:
                trim = trim * voxel_diag.item()
 
            mesh = trim_reconstructed_mesh(mesh, input_points.numpy(), trim)

        # Transform back to the original coordinate system.
        mesh.vertices = (mesh.vertices / scale.numpy()) + bb_center.numpy()

        return mesh