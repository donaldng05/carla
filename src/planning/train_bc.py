"""Config-driven BC training entrypoint with resumable checkpoints."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.bc_sequences import BehavioralCloningSequenceDataset, collate_bc_sequences
from src.data.carla_dataset import build_bc_preprocessing_config
from src.planning.bc_model import build_bc_model
from src.planning.bc_training import load_checkpoint, run_bc_epoch, save_checkpoint

LOGGER = logging.getLogger(__name__)


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the scalar/list subset used by ``configs/bc_v1.yaml`` without a hard dependency."""
    result: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split(":", 1)]
        try:
            value: Any = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            lowered = raw_value.lower()
            if lowered in {"true", "false"}:
                value = lowered == "true"
            else:
                try:
                    value = float(raw_value) if "." in raw_value else int(raw_value)
                except ValueError:
                    value = raw_value
        result[key] = value
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model: torch.nn.Module, train_loader: DataLoader[Any], val_loader: DataLoader[Any],
                config: dict[str, Any], *, device: torch.device,
                checkpoint_dir: str | Path) -> list[dict[str, float]]:
    """Train and checkpoint one model; datasets/loaders are injected for testability."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["num_epochs"]))
    start_epoch = 0
    best = float("inf")
    resume = config.get("resume_checkpoint")
    if resume:
        state = load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler,
                                map_location=device)
        start_epoch = int(state["epoch"]) + 1
        best = float(state.get("best_val_loss", best))
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, int(config["num_epochs"])):
        train = run_bc_epoch(
            model, train_loader, optimizer, device=device,
            steering_weight=float(config.get("steering_loss_weight", 2.0)),
            throttle_weight=float(config.get("throttle_loss_weight", 1.0)),
            augmentation_shift_pixels=int(config.get("augmentation_shift_pixels", 0)),
            augmentation_steering_gain=float(config.get("augmentation_steering_gain", 0.002)),
        )
        with torch.no_grad():
            validation = run_bc_epoch(
                model, val_loader, None, device=device,
                steering_weight=float(config.get("steering_loss_weight", 2.0)),
                throttle_weight=float(config.get("throttle_loss_weight", 1.0)),
            )
        scheduler.step()
        record = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train.items()},
                  **{f"val_{k}": v for k, v in validation.items()}}
        history.append(record)
        checkpoint_path = Path(checkpoint_dir) / f"ckpt_epoch_{epoch:03d}.pt"
        save_checkpoint(checkpoint_path, epoch=epoch, model=model, optimizer=optimizer,
                        scheduler=scheduler, best_val_loss=min(best, validation["loss"]),
                        config=config)
        if validation["loss"] < best:
            best = validation["loss"]
            save_checkpoint(Path(checkpoint_dir) / "best_checkpoint.pt", epoch=epoch,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            best_val_loss=best, config=config)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc_v1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.parse_args()
    raise SystemExit("Use train_model() from a notebook or project-specific dataset runner.")


if __name__ == "__main__":
    main()
