import torch
import argparse
from pathlib import Path
import point_cloud_utils as pcu

from vksr import Reconstructor


def main(args):
    Path(args.output_dir).mkdir(parents = True, exist_ok = True)

    input_points, input_normals = pcu.load_mesh_vn(args.point_cloud)
    
    input_points = torch.from_numpy(input_points)
    input_normals = torch.from_numpy(input_normals)

    reconstructor = Reconstructor(knn_searcher = args.knn_searcher,
                                  knn_searcher_kwargs = {'nlist': args.nlist, 'nprobe': args.nprobe},
                                  kernel = args.kernel,
                                  kernel_kwargs = {'order': args.order, 'h': args.h},
                                  use_abs_units = args.use_abs_units,
                                  device = args.device)

    mesh = reconstructor.reconstruct(input_points, input_normals,
                                     k = args.k, 
                                     reg_weight = args.reg_weight, 
                                     chunk_size = args.chunk_size,
                                     surf_eps = args.surf_eps, 
                                     voxel_resolution_0 = args.voxel_resolution_0,
                                     num_upsampling_steps = args.num_upsampling_steps, 
                                     trim = args.trim)
    
    mesh.export(f'{args.output_dir}/reconstruction.ply')
    print('Saved reconstructed mesh.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('point_cloud', type = str, 
                           help = 'Path to point cloud (with normals) that should be reconstructed.')
    argparser.add_argument('k', type = int,
                            help = 'Number of nearest neighbors.')
    argparser.add_argument('reg_weight', type = float,
                            help = 'Regularization weight.')
    argparser.add_argument('--output_dir', type = str, 
                           help = 'The directory to save the reconstruction to.',
                           default = './')
    argparser.add_argument('--chunk_size', type = int,
                            help = 'Number of query points to process in each chunk.',
                            default = 1000)
    argparser.add_argument('--surf_eps', type = float,
                            help = 'Offset distance along the normals.',
                            default = 0.05)
    argparser.add_argument('--voxel_resolution_0', type = int,
                            help = 'Initial voxel resolution for MISE algorithm.',
                            default = 32)
    argparser.add_argument('--num_upsampling_steps', type = int,
                            help = 'Number of upsampling steps for MISE algorithm.',
                            default = 3)
    argparser.add_argument('--trim', type = float,
                            help = 'Distance threshold (in voxels) used to trim away spurious geometry, if any.',
                            default = 0.0)
    argparser.add_argument('--knn_searcher', type = str,
                            help = "Nearest neighbor search strategy. Choose either 'approximate_faiss' for approximate" \
                            "nearest neighbors or 'exact_faiss' for exact nearest neighbor search.",
                            default = 'approximate_faiss')
    argparser.add_argument('--nlist', type = int,
                            help = "Number of centroids for the approximate nearest neighbor search (ignored if knn_searcher"
                            "is not 'approximate_faiss').",
                            default = 1000)
    argparser.add_argument('--nprobe', type = int,
                            help = "Number of probes for the approximate nearest neighbor search (ignored if knn_searcher"
                            "is not 'approximate_faiss').",
                            default = 10)
    argparser.add_argument('--kernel', type = str,
                            help = "Kernel function to use for reconstruction. As of now, only 'matern' is available.",
                            default = 'matern')
    argparser.add_argument('--order', type = str,
                            help = "Order of the Matérn kernel. Choose from '1/2' (Laplace), '3/2', '5/2', or 'inf' (Gaussian).",
                            default = '1/2')
    argparser.add_argument('--h', type = float,
                            help = 'Bandwidth parameter of the Matérn kernel.',
                            default = 1.0)
    argparser.add_argument('--use_abs_units', action = 'store_true',
                            help = 'Whether to use absolute units for surf_eps and trim. If not set, values are interpreted' \
                            'relative to the voxel size.')
    argparser.add_argument('--device', type = str,
                            help = "The device to use for reconstruction. Choose either 'cuda' or 'cpu'.",
                            default = 'cuda')

    args, _ = argparser.parse_known_args()

    main(args)