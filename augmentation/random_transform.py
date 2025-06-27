import torch
import torch.nn.functional as F
from typing import Union, Tuple


class RandTransform3D:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform with probability check"""
        if torch.rand(1).item() > self.prob:
            return x
        return self.apply_transform(x)

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate_random_params(self, batch_size: int, device: torch.device):
        """Generate random parameters for each item in batch"""
        raise NotImplementedError


class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


class RandAffine3D(RandTransform3D):
    def __init__(self,
                 translate_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                 range_x: Union[float, Tuple[float, float]] = 0.0,
                 range_y: Union[float, Tuple[float, float]] = 0.0,
                 range_z: Union[float, Tuple[float, float]] = 0.0,
                 prob: float = 0.5):
        super().__init__(prob)
        self.translate_range = self._process_translate_range(translate_range)
        self.range_x = self._process_range(range_x)
        self.range_y = self._process_range(range_y)
        self.range_z = self._process_range(range_z)

    def _process_translate_range(self, translate_range):
        if len(translate_range) == 3 and isinstance(translate_range[0], (int, float)):
            # Single values for each axis
            return [(-abs(t), abs(t)) for t in translate_range]
        else:
            # Range tuples for each axis
            return translate_range

    def _process_range(self, range_val):
        if isinstance(range_val, (int, float)):
            return (-abs(range_val), abs(range_val))
        return tuple(range_val)

    def generate_random_params(self, batch_size: int, device: torch.device):
        """Generate random rotation angles and translation for each batch item"""
        angles_x = torch.empty(batch_size, device=device).uniform_(*self.range_x)
        angles_y = torch.empty(batch_size, device=device).uniform_(*self.range_y)
        angles_z = torch.empty(batch_size, device=device).uniform_(*self.range_z)

        translations = []
        for i in range(3):  # x, y, z
            min_val, max_val = self.translate_range[i]
            trans = torch.empty(batch_size, device=device).uniform_(min_val, max_val)
            translations.append(trans)

        return torch.stack(translations, dim=1), angles_x, angles_y, angles_z

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch-wise random rotations"""
        batch_size = x.shape[0]
        device = x.device

        # Generate random parameters for each batch item
        translations, angles_x, angles_y, angles_z = self.generate_random_params(batch_size, device)

        # Apply rotations using affine grids
        return self._batch_affine_3d(x, translations, angles_x, angles_y, angles_z)

    def _batch_affine_3d(self, x: torch.Tensor, translations: torch.Tensor, angles_x: torch.Tensor,
                         angles_y: torch.Tensor, angles_z: torch.Tensor) -> torch.Tensor:
        """Efficient batch rotation using affine grids"""
        B, C, D, H, W = x.shape

        # Normalize translations to [-1, 1] range for grid_sample
        normalized_translations = translations.clone()
        normalized_translations[:, 0] /= (W - 1) / 2  # x translation
        normalized_translations[:, 1] /= (H - 1) / 2  # y translation
        normalized_translations[:, 2] /= (D - 1) / 2  # z translation

        # Create rotation matrices for each batch item
        rotation_matrices = self._create_rotation_matrices(angles_x, angles_y, angles_z)

        # Create affine grids
        affine_matrices = torch.zeros(B, 3, 4, device=x.device)
        affine_matrices[:, :3, :3] = rotation_matrices
        affine_matrices[:, :, 3] = normalized_translations  # Translation

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
