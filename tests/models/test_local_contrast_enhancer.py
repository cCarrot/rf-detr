"""Tests for the trainable local-contrast enhancer."""

import pytest
import torch

from rfdetr.models.local_contrast_enhancer import DifferentiableLocalContrastEnhancer


def test_local_contrast_enhancer_preserves_padded_pixels_and_backpropagates() -> None:
    enhancer = DifferentiableLocalContrastEnhancer(window_size=3)
    images = torch.rand(2, 3, 8, 8, requires_grad=True)
    padding_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    padding_mask[1, :, 6:] = True

    enhanced = enhancer(images, padding_mask)

    assert torch.equal(enhanced[1, :, :, 6:], images[1, :, :, 6:])
    (enhanced.square().mean()).backward()
    assert enhancer.strength.grad is not None
    assert enhancer.gain.grad is not None


def test_local_contrast_enhancer_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="odd integer"):
        DifferentiableLocalContrastEnhancer(window_size=4)
