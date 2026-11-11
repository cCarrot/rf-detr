# ------------------------------------------------------------------------
# Research extension: differentiable local-contrast enhancement.
# ------------------------------------------------------------------------
"""A small trainable image enhancer for joint sea-fog detection training."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DifferentiableLocalContrastEnhancer(nn.Module):
    """Boost local image detail with trainable, detection-driven parameters.

    The module operates on the same normalized tensors as RF-DETR. It uses a
    masked local mean and standard deviation so padded batch regions do not
    distort the valid image border. Detection losses backpropagate through this
    module when it is called from :class:`LWDETR`'s regular forward path.
    This is a local-contrast operator, not a physical atmospheric dehazer.
    """

    def __init__(self, window_size: int = 9) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        self.window_size = window_size
        self.strength = nn.Parameter(torch.tensor(0.5))
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Enhance valid pixels while preserving padded values unchanged."""
        pad = self.window_size // 2
        if padding_mask is None:
            valid = images.new_ones((images.shape[0], 1, *images.shape[-2:]))
        else:
            if padding_mask.shape != (images.shape[0], *images.shape[-2:]):
                raise ValueError("padding_mask must have shape [B,H,W]")
            valid = (~padding_mask).unsqueeze(1).to(dtype=images.dtype)

        count = F.avg_pool2d(valid, self.window_size, stride=1, padding=pad).clamp_min(1e-6)
        local_mean = F.avg_pool2d(images * valid, self.window_size, stride=1, padding=pad) / count
        local_sq_mean = F.avg_pool2d(images.square() * valid, self.window_size, stride=1, padding=pad) / count
        local_std = (local_sq_mean - local_mean.square()).clamp_min(1e-5).sqrt()
        detail = (images - local_mean) / (local_std + 0.1)
        alpha = self.strength.sigmoid()
        enhanced = images + alpha * self.gain.clamp(0.1, 3.0) * detail * 0.1
        return enhanced if padding_mask is None else torch.where(valid.bool(), enhanced, images)
