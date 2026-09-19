import numpy as np
import torch


class MISE:
    """
    GPU-capable reimplementation of the MISE (Multiresolution IsoSurface Extraction) algorithm proposed in:
        Mescheder et al., Occupancy Networks: Learning 3D Reconstruction in Function Space, CVPR'19.
    The reference implemenation is available at:
        https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/utils/libmise/mise.pyx.
    :param resolution_0: The resolution of the initial coarse grid
    :param depth: The number of upsampling steps
    :param threshold: Threshold for determining zero crossing within a voxel
    :param device: The device to use for computation
    :param value_dtype: The data type for voxel values
    """
    def __init__(self, resolution_0, depth, threshold, device = 'cuda', value_dtype = torch.float32):
        self.resolution_0 = resolution_0
        self.depth = depth
        self.threshold = threshold
         
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            print('CUDA requested but not available. Using CPU instead.')
        else:
            self.device = torch.device(device)

        self.value_dtype = value_dtype
        
        self.voxel_size_0 = 1 << self.depth
        self.resolution = self.resolution_0 * self.voxel_size_0
        self.grid_resolution = self.resolution + 1

        max_grid_key = (self.grid_resolution * self.grid_resolution * self.grid_resolution) - 1
        self.key_dtype = torch.int32 if max_grid_key <= torch.iinfo(torch.int32).max else torch.int64
        self.coord_dtype = torch.int16 if self.resolution <= torch.iinfo(torch.int16).max else torch.int32

        self.index_dtype = torch.int64
        self.point_index_dtype = torch.int32
        self.level_dtype = torch.int16
        self.activation_dtype = torch.int8

        self.numpy_value_dtype = self._numpy_dtype_for_torch_dtype(self.value_dtype)
        self.numpy_index_dtype = self._numpy_dtype_for_torch_dtype(self.index_dtype)

        self._grid_key_stride_y = self.grid_resolution * self.grid_resolution
        self._grid_key_stride_z = self.grid_resolution

        self._neighbor_offsets = self._make_coord_tensor(
            [[i, j, k] for i in (-1, 0) for j in (-1, 0) for k in (-1, 0)]
        )
        self._corner_offsets = self._make_coord_tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        )
        self._subdiv_offsets = self._make_coord_tensor(
            [[i, j, k] for i in (0, 1, 2) for j in (0, 1, 2) for k in (0, 1, 2)]
        )
        self._corner_offsets_i64 = self._corner_offsets.to(self.index_dtype)
        self._subdiv_offsets_i64 = self._subdiv_offsets.to(self.index_dtype)

        # Lower these two numbers to reduce memory footprint.
        self.known_point_chunk_size = 262_144 
        self.parent_chunk_size = 16_384 

        self._grid_points_count = 0
        self._known_point_count = 0
        self._voxel_count = 0

        self._grid_points_xyz_storage = torch.empty((0, 3), dtype = self.coord_dtype, device = self.device)
        self.grid_point_keys_sorted = torch.empty((0,), dtype = self.key_dtype, device = self.device)
        self.grid_point_sorted_indices = torch.empty((0,), dtype = self.point_index_dtype, device = self.device)
        self._grid_points_value_storage = torch.empty((0,), dtype = self.value_dtype, device = self.device)
        self._grid_points_known_storage = torch.empty((0,), dtype = torch.bool, device = self.device)
        self._known_point_indices_storage = torch.empty((0,), dtype = self.point_index_dtype, device = self.device)
      
        self._next_to_pos_storage = torch.empty((0,), dtype = self.activation_dtype, device = self.device)
        self._next_to_neg_storage = torch.empty((0,), dtype = self.activation_dtype, device = self.device)

        self._voxel_loc_storage = torch.empty((0, 3), dtype = self.coord_dtype, device = self.device)
        self._voxel_level_storage = torch.empty((0,), dtype = self.level_dtype, device = self.device)
        self._voxel_leaf_storage = torch.empty((0,), dtype = torch.bool, device = self.device)
        self._voxel_children_storage = torch.empty((0, 2, 2, 2), dtype = self.point_index_dtype, device = self.device)

        self._sync_views()
        self._init_coarse_grid()

    @torch.no_grad()
    def update(self, points, values):
        points_t = self._as_coord_tensor(points)
        values_t = self._as_value_tensor(values)

        point_indices = self._lookup_point_indices(points_t)
        if (point_indices == -1).any():
            raise ValueError('Point not in grid.')

        idx64 = self._as_index_tensor(point_indices)
        newly_known_mask = ~self.grid_points_known[idx64]
        if newly_known_mask.any():
            self._append_known_point_indices(point_indices[newly_known_mask])
        self.grid_points_value[idx64] = values_t
        self.grid_points_known[idx64] = True

        self._subdivide_voxels()

    @torch.no_grad()
    def query(self):
        return self.grid_points_xyz[~self.grid_points_known].to(self.index_dtype).cpu().numpy()

    @torch.no_grad()
    def to_dense(self):
        out = np.full(
            (self.grid_resolution, self.grid_resolution, self.grid_resolution),
            np.nan,
            dtype = self.numpy_value_dtype,
        )
        xyz_cpu = self.grid_points_xyz.to(self.index_dtype).cpu().numpy()
        val_cpu = self.grid_points_value.cpu().numpy().astype(self.numpy_value_dtype, copy = False)

        out[xyz_cpu[:, 0], xyz_cpu[:, 1], xyz_cpu[:, 2]] = val_cpu

        self._forward_fill_nan_numpy(out, axis = 0)
        self._forward_fill_nan_numpy(out, axis = 1)
        self._forward_fill_nan_numpy(out, axis = 2)
        return out

    @torch.no_grad()
    def get_points(self):
        return (
            self.grid_points_xyz.to(self.index_dtype).cpu().numpy(),
            self.grid_points_value.cpu().numpy().astype(self.numpy_value_dtype, copy = False),
        )

    @torch.no_grad()
    def _init_coarse_grid(self):
        coarse_axis = self._coord_range(self.resolution_0 + 1)
        coarse_coords = torch.stack(
            torch.meshgrid(coarse_axis, coarse_axis, coarse_axis, indexing = 'ij'),
            dim = -1,
        ).reshape(-1, 3)
        coarse_points = coarse_coords * self.voxel_size_0
        self._add_grid_points_batch(coarse_points)

        coarse_voxel_axis = self._coord_range(self.resolution_0)
        coarse_voxel_coords = torch.stack(
            torch.meshgrid(coarse_voxel_axis, coarse_voxel_axis, coarse_voxel_axis, indexing = 'ij'),
            dim = -1,
        ).reshape(-1, 3)
        coarse_locs = coarse_voxel_coords * self.voxel_size_0

        self._append_voxels(
            coarse_locs,
            torch.zeros((coarse_locs.shape[0],), dtype = self.level_dtype, device = self.device),
            torch.ones((coarse_locs.shape[0],), dtype = torch.bool, device = self.device),
            torch.full((coarse_locs.shape[0], 2, 2, 2), -1, dtype = self.point_index_dtype, device = self.device),
        )

    @torch.no_grad()
    def _subdivide_voxels(self):
        known_idx = self.known_point_indices
        if known_idx.numel() == 0:
            return

        voxel_count = self.voxel_loc.shape[0]
        self._ensure_activation_capacity(voxel_count)
        next_to_pos = self._next_to_pos_storage[:voxel_count]
        next_to_neg = self._next_to_neg_storage[:voxel_count]
        next_to_pos.zero_()
        next_to_neg.zero_()

        for start in range(0, known_idx.shape[0], self.known_point_chunk_size):
            stop = min(start + self.known_point_chunk_size, known_idx.shape[0])
            chunk_idx = known_idx[start:stop].to(self.index_dtype)

            known_xyz = self.grid_points_xyz[chunk_idx]
            known_values = self.grid_points_value[chunk_idx]

            neighbor_xyz = known_xyz[:, None, :] + self._neighbor_offsets[None, :, :]
            valid_mask = self._valid_voxel_mask(neighbor_xyz)
            if not valid_mask.any():
                continue

            flat_valid = valid_mask.reshape(-1)
            neighbor_xyz = neighbor_xyz.reshape(-1, 3)[flat_valid]
            neighbor_values = known_values.repeat_interleave(8)[flat_valid]

            neighbor_xyz_unique, inverse = torch.unique(neighbor_xyz, dim = 0, return_inverse = True)
            voxel_idx_unique = self._get_voxel_idx_batch(neighbor_xyz_unique)
            voxel_idx = voxel_idx_unique[inverse]

            next_to_pos.scatter_reduce_(
                0,
                voxel_idx,
                (neighbor_values >= self.threshold).to(self.activation_dtype),
                reduce = 'amax',
                include_self = True,
            )
            next_to_neg.scatter_reduce_(
                0,
                voxel_idx,
                (neighbor_values <= self.threshold).to(self.activation_dtype),
                reduce = 'amax',
                include_self = True,
            )

        active_mask = self.voxel_leaf & (self.voxel_level < self.depth) & next_to_pos.bool() & next_to_neg.bool()
        parent_idx = active_mask.nonzero(as_tuple = False).squeeze(1)
        if parent_idx.numel() == 0:
            return

        self._subdivide_voxels_batch(parent_idx)

    @torch.no_grad()
    def _subdivide_voxels_batch(self, parent_idx):
        parent_idx = self._as_index_tensor(parent_idx)

        for start in range(0, parent_idx.shape[0], self.parent_chunk_size):
            stop = min(start + self.parent_chunk_size, parent_idx.shape[0])
            parent_chunk = parent_idx[start:stop]
            num_parents = parent_chunk.shape[0]

            parent_locs = self.voxel_loc[parent_chunk].to(self.index_dtype)
            parent_levels = self.voxel_level[parent_chunk].to(self.index_dtype)
            new_levels = (parent_levels + 1).to(self.level_dtype)
            new_size = (1 << (self.depth - (parent_levels + 1))).to(self.index_dtype)

            child_offsets = self._corner_offsets_i64
            child_locs = (
                parent_locs[:, None, :] + child_offsets[None, :, :] * new_size[:, None, None]
            ).reshape(-1, 3)

            subdiv_offsets = self._subdiv_offsets_i64
            subdiv_xyz = (
                parent_locs[:, None, :] + subdiv_offsets[None, :, :] * new_size[:, None, None]
            ).reshape(-1, 3).to(self.coord_dtype)

            subdiv_keys = self._point_keys(subdiv_xyz)
            subdiv_existing = self._lookup_point_indices(subdiv_xyz)
            missing_mask = subdiv_existing == -1
            if missing_mask.any():
                missing_idx = missing_mask.nonzero(as_tuple = False).squeeze(1)
                missing_keys = subdiv_keys[missing_idx]

                key_sort = torch.argsort(missing_keys, stable = True)
                sorted_keys = missing_keys[key_sort]
                unique_first = torch.ones_like(sorted_keys, dtype = torch.bool)
                if sorted_keys.numel() > 1:
                    unique_first[1:] = sorted_keys[1:] != sorted_keys[:-1]

                first_missing_idx = missing_idx[key_sort[unique_first]]
                first_missing_idx = torch.sort(first_missing_idx).values
                self._add_grid_points_batch(subdiv_xyz[first_missing_idx])

            first_child_idx = self.voxel_loc.shape[0]
            child_ids = torch.arange(
                first_child_idx,
                first_child_idx + (8 * num_parents),
                dtype = self.point_index_dtype,
                device = self.device,
            ).view(num_parents, 2, 2, 2)

            self._ensure_voxel_capacity(8 * num_parents)
            self.voxel_leaf[parent_chunk] = False
            self.voxel_children[parent_chunk] = child_ids

            self._append_voxels(
                child_locs.to(self.coord_dtype),
                new_levels.repeat_interleave(8),
                torch.ones((8 * num_parents,), dtype = torch.bool, device = self.device),
                torch.full((8 * num_parents, 2, 2, 2), -1, dtype = self.point_index_dtype, device = self.device),
            )

    @torch.no_grad()
    def _get_voxel_idx_batch(self, locs):
        locs = self._as_index_tensor(locs)

        coarse = locs >> self.depth
        idx = (
            coarse[:, 0] * (self.resolution_0 * self.resolution_0)
            + coarse[:, 1] * self.resolution_0
            + coarse[:, 2]
        ).to(self.index_dtype)

        loc_rel = locs - (coarse << self.depth)
        voxel_size = torch.full((locs.shape[0],), self.voxel_size_0, dtype=self.index_dtype, device=self.device)

        for _ in range(self.depth):
            leaf_mask = self.voxel_leaf[idx]
            if leaf_mask.all():
                break

            need_descend = ~leaf_mask
            child_size = voxel_size >> 1
            child_offset = (loc_rel >= child_size[:, None]).to(self.index_dtype)

            next_idx = idx.clone()
            next_idx[need_descend] = self.voxel_children[
                idx[need_descend],
                child_offset[need_descend, 0],
                child_offset[need_descend, 1],
                child_offset[need_descend, 2],
            ].to(self.index_dtype)

            loc_rel = loc_rel - child_offset * child_size[:, None]
            voxel_size = torch.where(need_descend, child_size, voxel_size)
            idx = next_idx

        return idx

    def _sync_views(self):
        self.grid_points_xyz = self._grid_points_xyz_storage[:self._grid_points_count]
        self.grid_points_value = self._grid_points_value_storage[:self._grid_points_count]
        self.grid_points_known = self._grid_points_known_storage[:self._grid_points_count]
        self.known_point_indices = self._known_point_indices_storage[:self._known_point_count]

        self.voxel_loc = self._voxel_loc_storage[:self._voxel_count]
        self.voxel_level = self._voxel_level_storage[:self._voxel_count]
        self.voxel_leaf = self._voxel_leaf_storage[:self._voxel_count]
        self.voxel_children = self._voxel_children_storage[:self._voxel_count]

    def _ensure_grid_point_capacity(self, additional_count):
        required = self._grid_points_count + additional_count
        current = self._grid_points_xyz_storage.shape[0]
        new_capacity = self._next_capacity(current, required)
        if new_capacity == current:
            return

        xyz_storage = torch.empty((new_capacity, 3), dtype = self.coord_dtype, device = self.device)
        value_storage = torch.empty((new_capacity,), dtype = self.value_dtype, device = self.device)
        known_storage = torch.empty((new_capacity,), dtype = torch.bool, device = self.device)
        if self._grid_points_count != 0:
            xyz_storage[:self._grid_points_count] = self.grid_points_xyz
            value_storage[:self._grid_points_count] = self.grid_points_value
            known_storage[:self._grid_points_count] = self.grid_points_known

        self._grid_points_xyz_storage = xyz_storage
        self._grid_points_value_storage = value_storage
        self._grid_points_known_storage = known_storage
        self._sync_views()

    def _ensure_activation_capacity(self, required_count):
        current = self._next_to_pos_storage.shape[0]
        new_capacity = self._next_capacity(current, required_count)
        if new_capacity == current:
            return

        self._next_to_pos_storage = torch.empty((new_capacity,), dtype = self.activation_dtype, device = self.device)
        self._next_to_neg_storage = torch.empty((new_capacity,), dtype = self.activation_dtype, device = self.device)

    def _ensure_known_point_capacity(self, additional_count):
        required = self._known_point_count + additional_count
        current = self._known_point_indices_storage.shape[0]
        new_capacity = self._next_capacity(current, required)
        if new_capacity == current:
            return

        storage = torch.empty((new_capacity,), dtype = self.point_index_dtype, device = self.device)
        if self._known_point_count != 0:
            storage[:self._known_point_count] = self.known_point_indices
        self._known_point_indices_storage = storage
        self._sync_views()

    def _append_known_point_indices(self, indices):
        if indices.numel() == 0:
            return

        indices = self._as_point_index_tensor(indices)
        old_count = self._known_point_count
        add_count = indices.shape[0]
        self._ensure_known_point_capacity(add_count)
        self._known_point_indices_storage[old_count:old_count + add_count] = indices
        self._known_point_count = old_count + add_count

        self._sync_views()

    def _ensure_voxel_capacity(self, additional_count):
        required = self._voxel_count + additional_count
        current = self._voxel_loc_storage.shape[0]
        new_capacity = self._next_capacity(current, required)
        if new_capacity == current:
            return

        loc_storage = torch.empty((new_capacity, 3), dtype = self.coord_dtype, device = self.device)
        level_storage = torch.empty((new_capacity,), dtype = self.level_dtype, device = self.device)
        leaf_storage = torch.empty((new_capacity,), dtype = torch.bool, device = self.device)
        children_storage = torch.empty((new_capacity, 2, 2, 2), dtype = self.point_index_dtype, device = self.device)
        if self._voxel_count != 0:
            loc_storage[:self._voxel_count] = self.voxel_loc
            level_storage[:self._voxel_count] = self.voxel_level
            leaf_storage[:self._voxel_count] = self.voxel_leaf
            children_storage[:self._voxel_count] = self.voxel_children

        self._voxel_loc_storage = loc_storage
        self._voxel_level_storage = level_storage
        self._voxel_leaf_storage = leaf_storage
        self._voxel_children_storage = children_storage
        self._sync_views()

    def _append_voxels(self, locs, levels, leaf, children):
        if locs.numel() == 0:
            return

        count = locs.shape[0]
        self._ensure_voxel_capacity(count)
        start = self._voxel_count
        end = start + count
        self._voxel_loc_storage[start:end] = self._as_coord_tensor(locs)
        self._voxel_level_storage[start:end] = levels.to(self.level_dtype)
        self._voxel_leaf_storage[start:end] = leaf
        self._voxel_children_storage[start:end] = children.to(self.point_index_dtype)
        self._voxel_count = end
        self._sync_views()

    def _add_grid_points_batch(self, xyz):
        if xyz.numel() == 0:
            return

        xyz = self._as_coord_tensor(xyz)
        keys = self._point_keys(xyz)

        start = self._grid_points_count
        end = start + xyz.shape[0]
        indices = torch.arange(start, end, dtype = self.point_index_dtype, device = self.device)

        self._ensure_grid_point_capacity(xyz.shape[0])
        self._grid_points_xyz_storage[start:end] = xyz
        self._grid_points_value_storage[start:end] = 0.0
        self._grid_points_known_storage[start:end] = False
        self._grid_points_count = end
        self._sync_views()

        self._merge_point_key_index(keys, indices)

    def _merge_point_key_index(self, new_keys, new_indices):
        if new_keys.numel() == 0:
            return

        sort_order = torch.argsort(new_keys)
        new_keys_sorted = new_keys[sort_order]
        new_indices_sorted = self._as_point_index_tensor(new_indices[sort_order])

        if self.grid_point_keys_sorted.numel() == 0:
            self.grid_point_keys_sorted = new_keys_sorted
            self.grid_point_sorted_indices = new_indices_sorted
            return

        old_keys = self.grid_point_keys_sorted
        old_indices = self.grid_point_sorted_indices

        insert_pos = torch.searchsorted(old_keys, new_keys_sorted)
        merged_insert_pos = insert_pos + torch.arange(new_keys_sorted.shape[0], dtype = self.index_dtype, device = self.device)
        merged_size = old_keys.shape[0] + new_keys_sorted.shape[0]

        insert_mask = torch.zeros((merged_size,), dtype = torch.bool, device = self.device)
        insert_mask[merged_insert_pos] = True

        merged_keys = torch.empty((merged_size,), dtype = self.key_dtype, device = self.device)
        merged_indices = torch.empty((merged_size,), dtype = self.point_index_dtype, device = self.device)
        merged_keys[insert_mask] = new_keys_sorted
        merged_indices[insert_mask] = new_indices_sorted
        merged_keys[~insert_mask] = old_keys
        merged_indices[~insert_mask] = old_indices

        self.grid_point_keys_sorted = merged_keys
        self.grid_point_sorted_indices = merged_indices

    def _lookup_point_indices(self, xyz):
        keys = self._point_keys(xyz)
        if self.grid_point_keys_sorted.numel() == 0:
            return torch.full((keys.shape[0],), -1, dtype = self.point_index_dtype, device = self.device)

        pos = torch.searchsorted(self.grid_point_keys_sorted, keys)
        valid = pos < self.grid_point_keys_sorted.shape[0]
        out = torch.full((keys.shape[0],), -1, dtype = self.point_index_dtype, device = self.device)

        if valid.any():
            valid_pos = pos[valid]
            match = self.grid_point_keys_sorted[valid_pos] == keys[valid]
            if match.any():
                valid_indices = valid.nonzero(as_tuple = False).squeeze(1)[match]
                out[valid_indices] = self.grid_point_sorted_indices[valid_pos[match]]

        return out

    def _point_keys(self, xyz):
        xyz_keys = xyz.to(self.key_dtype)
        return (
            xyz_keys[:, 0] * self._grid_key_stride_y
            + xyz_keys[:, 1] * self._grid_key_stride_z
            + xyz_keys[:, 2]
        )

    def _valid_voxel_mask(self, xyz):
        return (
            (xyz[..., 0] >= 0)
            & (xyz[..., 0] < self.resolution)
            & (xyz[..., 1] >= 0)
            & (xyz[..., 1] < self.resolution)
            & (xyz[..., 2] >= 0)
            & (xyz[..., 2] < self.resolution)
        )

    def _next_capacity(self, current_capacity, required_capacity):
        if current_capacity >= required_capacity:
            return current_capacity
        if current_capacity == 0:
            return max(required_capacity, 16)
        growth = max(current_capacity // 4, 1024)
        return max(required_capacity, current_capacity + growth)

    def _coord_range(self, stop):
        return torch.arange(stop, dtype = self.coord_dtype, device = self.device)

    def _as_coord_tensor(self, x):
        return torch.as_tensor(x, dtype = self.coord_dtype, device = self.device)

    def _as_value_tensor(self, x):
        return torch.as_tensor(x, dtype = self.value_dtype, device = self.device)

    def _as_index_tensor(self, x):
        return torch.as_tensor(x, dtype = self.index_dtype, device = self.device)

    def _as_point_index_tensor(self, x):
        return torch.as_tensor(x, dtype = self.point_index_dtype, device = self.device)

    def _make_coord_tensor(self, values):
        return torch.tensor(values, dtype = self.coord_dtype, device = self.device)


    @staticmethod
    def _numpy_dtype_for_torch_dtype(torch_dtype):
        if torch_dtype == torch.float64:
            return np.float64
        if torch_dtype == torch.float32:
            return np.float32
        if torch_dtype == torch.float16:
            return np.float16
        if torch_dtype == torch.bfloat16:
            return np.float32
        if torch_dtype == torch.int64:
            return np.int64
        if torch_dtype == torch.int32:
            return np.int32
        if torch_dtype == torch.int16:
            return np.int16
        if torch_dtype == torch.int8:
            return np.int8
        if torch_dtype == torch.uint8:
            return np.uint8
        if torch_dtype == torch.bool:
            return np.bool_
        raise ValueError(f'Unsupported torch dtype for numpy conversion: {torch_dtype}.')

    @staticmethod
    def _forward_fill_nan_numpy(array, axis):
        for idx in range(1, array.shape[axis]):
            cur_slice = [slice(None)] * array.ndim
            prev_slice = [slice(None)] * array.ndim
            cur_slice[axis] = idx
            prev_slice[axis] = idx - 1
            cur = array[tuple(cur_slice)]
            prev = array[tuple(prev_slice)]
            np.copyto(cur, prev, where = np.isnan(cur))