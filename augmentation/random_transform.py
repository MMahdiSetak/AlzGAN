import torch


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
