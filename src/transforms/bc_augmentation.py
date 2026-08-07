"""Training-only image perturbations for behavioral cloning."""

from __future__ import annotations

import torch


def shift_sequence_horizontally(
    frames: torch.Tensor, *, shift_pixels: int, steering_gain: float = 0.002
) -> tuple[torch.Tensor, float]:
    """Shift a ``(T,C,H,W)`` sequence and return its corrective steering offset."""
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape (T,C,H,W), got {tuple(frames.shape)}")
    if not isinstance(shift_pixels, int):
        raise TypeError("shift_pixels must be an integer")
    width = frames.shape[-1]
    if abs(shift_pixels) >= width:
        raise ValueError("shift_pixels must be smaller than image width")
    shifted = torch.zeros_like(frames)
    if shift_pixels >= 0:
        shifted[..., shift_pixels:] = frames[..., : width - shift_pixels]
    else:
        amount = -shift_pixels
        shifted[..., : width - amount] = frames[..., amount:]
    return shifted, float(-shift_pixels * steering_gain)
