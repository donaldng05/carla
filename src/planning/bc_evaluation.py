"""Offline behavioral-cloning metrics."""

from __future__ import annotations

from typing import Iterable

import torch


def action_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute MAE/RMSE for ``[steering, throttle]`` actions."""
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and targets must have shape (N,2)")
    error = predictions.detach().float() - targets.detach().float()
    return {"sample_count": float(predictions.shape[0]),
            "steering_mae": float(error[:, 0].abs().mean()),
            "throttle_mae": float(error[:, 1].abs().mean()),
            "steering_rmse": float(error[:, 0].pow(2).mean().sqrt()),
            "throttle_rmse": float(error[:, 1].pow(2).mean().sqrt())}


def aggregate_action_metrics(batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    """Aggregate predictions before calculating metrics to avoid batch-size bias."""
    predictions, targets = [], []
    for prediction, target in batches:
        predictions.append(prediction)
        targets.append(target)
    if not predictions:
        raise ValueError("at least one prediction batch is required")
    return action_metrics(torch.cat(predictions), torch.cat(targets))
