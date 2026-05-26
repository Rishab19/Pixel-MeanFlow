import torch
import math


class Spiral2DDataset:
    def __init__(self, device="cuda"):
        self.obs_dim = 2
        self.device = device

    def get_batch(self, n_samples):
        # Sample parameter along the spiral
        t = torch.rand(n_samples, device=self.device)

        # Spiral angle
        angle = t * 3 * math.pi  # 1.5 turns

        # Radius grows with angle
        r = 0.5 + t

        # 2D spiral coordinates
        x = r * torch.cos(angle)
        y = r * torch.sin(angle)

        # Return raw coordinates
        return torch.stack([x, y], dim=-1)

    def project_back(self, x):
        # Identity since no projection
        return x