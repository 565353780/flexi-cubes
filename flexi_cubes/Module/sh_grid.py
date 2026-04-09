import torch
import torch.nn as nn

from flexi_cubes.Module.sh_utils import eval_sh, SH2RGB


class SHGrid(nn.Module):
    """
    Independent SH-grid sampler.
    Does NOT own learnable parameters.
    External sh_coeff should be passed in.

    Coordinate convention matches FlexiCubes.construct_voxel_grid:
        grid domain is [-0.5, 0.5]
    """

    def __init__(self, resolution, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.resolution = self._to_res_tuple(resolution)
        self.rx, self.ry, self.rz = self.resolution

        self.nx = self.rx + 1
        self.ny = self.ry + 1
        self.nz = self.rz + 1
        self.num_vertices = self.nx * self.ny * self.nz

        grid_vertices, cube_idx = self.construct_voxel_grid(self.resolution)
        self.register_buffer("grid_vertices", grid_vertices)   # [Nv, 3]
        self.register_buffer("cube_idx", cube_idx)             # [Nc, 8]

    def _to_res_tuple(self, resolution):
        if isinstance(resolution, int):
            return (resolution, resolution, resolution)
        assert isinstance(resolution, (tuple, list)) and len(resolution) == 3
        return tuple(int(r) for r in resolution)

    def construct_voxel_grid(self, resolution):
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)

        rx, ry, rz = resolution
        device = self.device

        cube_corners = torch.tensor(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ],
            dtype=torch.float32,
            device=device
        )

        base_cube_f = torch.arange(8, device=device)
        voxel_grid_template = torch.ones((rx, ry, rz), device=device)
        resolution_t = torch.tensor([[rx, ry, rz]], dtype=torch.float32, device=device)

        coords = torch.nonzero(voxel_grid_template).float() / resolution_t
        verts = (cube_corners.unsqueeze(0) / resolution_t + coords.unsqueeze(1)).reshape(-1, 3)
        cubes = (base_cube_f.unsqueeze(0) + torch.arange(coords.shape[0], device=device).unsqueeze(1) * 8).reshape(-1)

        verts_rounded = torch.round(verts * 1e5) / 1e5
        verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)

        return verts_unique - 0.5, cubes

    def _flatten_idx(self, ix, iy, iz):
        return ix * (self.ny * self.nz) + iy * self.nz + iz

    def __call__(self, points: torch.Tensor, sh_coeff: torch.Tensor, detach_points: bool = True):
        """
        Args:
            points: [N, 3], coordinates in [-0.5, 0.5]
            sh_coeff: [Nv, C, SH]
                Nv = (rx+1)*(ry+1)*(rz+1)

        Returns:
            query_sh_coeff: [N, C, SH]
                trilinearly interpolated SH coefficients at query points
        """
        if detach_points:
            points = points.detach()

        assert torch.is_tensor(points), "'points' should be a tensor"
        assert points.ndim == 2 and points.shape[1] == 3, \
            "'points' should have shape [N, 3]"

        assert torch.is_tensor(sh_coeff), "'sh_coeff' should be a tensor"
        assert sh_coeff.ndim == 3, \
            "'sh_coeff' should have shape [Nv, C, SH]"
        assert sh_coeff.shape[0] == self.num_vertices, \
            f"'sh_coeff' first dim should be {self.num_vertices}, got {sh_coeff.shape[0]}"

        device = sh_coeff.device
        points = points.to(device)

        eps = 1e-6

        # map [-0.5, 0.5] -> [0, res]
        xyz = points + 0.5
        xyz_x = xyz[:, 0].clamp(0.0, 1.0 - eps) * self.rx
        xyz_y = xyz[:, 1].clamp(0.0, 1.0 - eps) * self.ry
        xyz_z = xyz[:, 2].clamp(0.0, 1.0 - eps) * self.rz

        ix0 = torch.floor(xyz_x).long()
        iy0 = torch.floor(xyz_y).long()
        iz0 = torch.floor(xyz_z).long()

        ix1 = (ix0 + 1).clamp(max=self.rx)
        iy1 = (iy0 + 1).clamp(max=self.ry)
        iz1 = (iz0 + 1).clamp(max=self.rz)

        tx = (xyz_x - ix0.float()).view(-1, 1, 1)
        ty = (xyz_y - iy0.float()).view(-1, 1, 1)
        tz = (xyz_z - iz0.float()).view(-1, 1, 1)

        idx000 = self._flatten_idx(ix0, iy0, iz0)
        idx100 = self._flatten_idx(ix1, iy0, iz0)
        idx010 = self._flatten_idx(ix0, iy1, iz0)
        idx110 = self._flatten_idx(ix1, iy1, iz0)
        idx001 = self._flatten_idx(ix0, iy0, iz1)
        idx101 = self._flatten_idx(ix1, iy0, iz1)
        idx011 = self._flatten_idx(ix0, iy1, iz1)
        idx111 = self._flatten_idx(ix1, iy1, iz1)

        c000 = sh_coeff[idx000]   # [N, C, SH]
        c100 = sh_coeff[idx100]
        c010 = sh_coeff[idx010]
        c110 = sh_coeff[idx110]
        c001 = sh_coeff[idx001]
        c101 = sh_coeff[idx101]
        c011 = sh_coeff[idx011]
        c111 = sh_coeff[idx111]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx

        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty

        query_sh_coeff = c0 * (1.0 - tz) + c1 * tz
        return query_sh_coeff
    
    