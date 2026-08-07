"""Loss, checkpoint, and small training-loop primitives for BC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.planning.bc_evaluation import action_metrics
from src.transforms.bc_augmentation import shift_sequence_horizontally


def weighted_bc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    steering_weight: float = 2.0,
    throttle_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted action loss and detached component metrics."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must both have shape (B,2)")
    errors = (prediction - target).pow(2)
    per_sample = steering_weight * errors[:, 0] + throttle_weight * errors[:, 1]
    if sample_weight is not None:
        if sample_weight.shape != (prediction.shape[0],):
            raise ValueError("sample_weight must have shape (B,)")
        loss = (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()
    return loss, {
        "steering_loss": errors[:, 0].mean().detach(),
        "throttle_loss": errors[:, 1].mean().detach(),
        "loss": loss.detach(),
    }


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_loss: float,
    config: dict[str, Any],
) -> Path:
    """Save resumable training state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer/scheduler state."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def run_bc_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    steering_weight: float = 2.0,
    throttle_weight: float = 1.0,
    augmentation_shift_pixels: int = 0,
    augmentation_steering_gain: float = 0.002,
) -> dict[str, float]:
    """Run one train/eval epoch over batches produced by ``collate_bc_sequences``."""
    training = optimizer is not None
    model.train(training)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    for batch in loader:
        video = batch["rgb"].to(device)
        target = batch["bc_target"].to(device)
        if training and augmentation_shift_pixels:
            shifted = []
            adjusted = target.clone()
            for index, sequence in enumerate(video):
                shift = int(
                    torch.randint(
                        -augmentation_shift_pixels, augmentation_shift_pixels + 1, ()
                    ).item()
                )
                sequence, correction = shift_sequence_horizontally(
                    sequence, shift_pixels=shift, steering_gain=augmentation_steering_gain
                )
                shifted.append(sequence)
                adjusted[index, 0] += correction
            video, target = torch.stack(shifted), adjusted
        with torch.set_grad_enabled(training):
            prediction = model(video)
            loss, _ = weighted_bc_loss(
                prediction,
                target,
                sample_weight=batch["sample_weight"].to(device),
                steering_weight=steering_weight,
                throttle_weight=throttle_weight,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
    metrics = action_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics
