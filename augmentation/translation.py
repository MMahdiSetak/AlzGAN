import torch
import torch.nn.functional as F
from typing import Union, Tuple

from augmentation.random_transform import RandTransform3D


class RandTranslate3D(RandTransform3D):
    def __init__(self,
                 translate_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                 prob: float = 0.5):
        super().__init__(prob)
        self.translate_range = self._process_translate_range(translate_range)

    def _process_translate_range(self, translate_range):
        if len(translate_range) == 3 and isinstance(translate_range[0], (int, float)):
            # Single values for each axis
            return [(-abs(t), abs(t)) for t in translate_range]
        else:
            # Range tuples for each axis
            return translate_range

    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch-wise random translations"""
        batch_size = x.shape[0]
        device = x.device

        # Generate random translations for each batch item
        translations = self.generate_random_params(batch_size, device)

        return self._batch_translate_3d(x, translations)

    def generate_random_params(self, batch_size: int, device: torch.device):
        """Generate random translation parameters"""
        translations = []
        for i in range(3):  # x, y, z
            min_val, max_val = self.translate_range[i]
            trans = torch.empty(batch_size, device=device).uniform_(min_val, max_val)
            translations.append(trans)
        return torch.stack(translations, dim=1)  # [B, 3]

    def _batch_translate_3d(self, x: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
        """Efficient batch translation using affine grids"""
        B, C, D, H, W = x.shape

        # Normalize translations to [-1, 1] range for grid_sample
        normalized_translations = translations.clone()
        normalized_translations[:, 0] /= (W - 1) / 2  # x translation
        normalized_translations[:, 1] /= (H - 1) / 2  # y translation
        normalized_translations[:, 2] /= (D - 1) / 2  # z translation

        # Create affine transformation matrices
        affine_matrices = torch.zeros(B, 3, 4, device=x.device)
        affine_matrices[:, 0, 0] = 1  # Identity rotation
        affine_matrices[:, 1, 1] = 1
        affine_matrices[:, 2, 2] = 1
        affine_matrices[:, :, 3] = normalized_translations  # Translation

        # Generate sampling grids and apply transformation
        grid = F.affine_grid(affine_matrices, (B, C, D, H, W), align_corners=False)
        translated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return translated
