import torch
import torch.nn.functional as F
from typing import Union, Tuple

from augmentation.random_transform import RandTransform3D

class RandRotate3D(RandTransform3D):
    def __init__(self,
                 range_x: Union[float, Tuple[float, float]] = 0.0,
                 range_y: Union[float, Tuple[float, float]] = 0.0,
                 range_z: Union[float, Tuple[float, float]] = 0.0,
                 prob: float = 0.5):
        super().__init__(prob)
        self.range_x = self._process_range(range_x)
        self.range_y = self._process_range(range_y)
        self.range_z = self._process_range(range_z)

    def _process_range(self, range_val):
        if isinstance(range_val, (int, float)):
            return (-abs(range_val), abs(range_val))
        return tuple(range_val)

    def generate_random_params(self, batch_size: int, device: torch.device):
        """Generate random rotation angles for each batch item"""
        angles_x = torch.empty(batch_size, device=device).uniform_(*self.range_x)
        angles_y = torch.empty(batch_size, device=device).uniform_(*self.range_y)
        angles_z = torch.empty(batch_size, device=device).uniform_(*self.range_z)
        return angles_x, angles_y, angles_z

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch-wise random rotations"""
        batch_size = x.shape[0]
        device = x.device

        # Generate random parameters for each batch item
        angles_x, angles_y, angles_z = self.generate_random_params(batch_size, device)

        # Apply rotations using affine grids
        return self._batch_rotate_3d(x, angles_x, angles_y, angles_z)

    def _batch_rotate_3d(self, x: torch.Tensor, angles_x: torch.Tensor,
                         angles_y: torch.Tensor, angles_z: torch.Tensor) -> torch.Tensor:
        """Efficient batch rotation using affine grids"""
        B, C, D, H, W = x.shape

        # Create rotation matrices for each batch item
        rotation_matrices = self._create_rotation_matrices(angles_x, angles_y, angles_z)

        # Create affine grids
        affine_matrices = torch.zeros(B, 3, 4, device=x.device)
        affine_matrices[:, :3, :3] = rotation_matrices

        # Generate sampling grids
        grid = F.affine_grid(affine_matrices, (B, C, D, H, W), align_corners=False)

        # Apply transformation
        rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return rotated

    def _create_rotation_matrices(self, angles_x: torch.Tensor,
                                  angles_y: torch.Tensor,
                                  angles_z: torch.Tensor) -> torch.Tensor:
        """Create 3D rotation matrices for batch"""
        batch_size = angles_x.shape[0]
        device = angles_x.device

        # Create individual rotation matrices
        cos_x, sin_x = torch.cos(angles_x), torch.sin(angles_x)
        cos_y, sin_y = torch.cos(angles_y), torch.sin(angles_y)
        cos_z, sin_z = torch.cos(angles_z), torch.sin(angles_z)

        # Rotation around X-axis
        Rx = torch.zeros(batch_size, 3, 3, device=device)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos_x
        Rx[:, 1, 2] = -sin_x
        Rx[:, 2, 1] = sin_x
        Rx[:, 2, 2] = cos_x

        # Rotation around Y-axis
        Ry = torch.zeros(batch_size, 3, 3, device=device)
        Ry[:, 0, 0] = cos_y
        Ry[:, 0, 2] = sin_y
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -sin_y
        Ry[:, 2, 2] = cos_y

        # Rotation around Z-axis
        Rz = torch.zeros(batch_size, 3, 3, device=device)
        Rz[:, 0, 0] = cos_z
        Rz[:, 0, 1] = -sin_z
        Rz[:, 1, 0] = sin_z
        Rz[:, 1, 1] = cos_z
        Rz[:, 2, 2] = 1

        # Combined rotation: Rz * Ry * Rx
        R = torch.bmm(torch.bmm(Rz, Ry), Rx)
        return R
