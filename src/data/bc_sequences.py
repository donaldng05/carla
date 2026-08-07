"""Run-safe temporal windows for behavioral cloning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from src.data.carla_dataset import CarlaDataPreprocessingConfig, CarlaMultimodalDataset


class BehavioralCloningSequenceDataset(Dataset[dict[str, Any]]):
    """Build past-only windows without crossing recording-run boundaries."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        sequence_length: int = 4,
        preprocessing: CarlaDataPreprocessingConfig | None = None,
        require_contiguous_frames: bool = True,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.samples = samples
        self.sequence_length = sequence_length
        self.base = CarlaMultimodalDataset(samples, preprocessing=preprocessing)
        self.window_indices = self._build_window_indices(require_contiguous_frames)

    def _build_window_indices(self, require_contiguous_frames: bool) -> list[tuple[int, ...]]:
        windows: list[tuple[int, ...]] = []
        for end in range(self.sequence_length - 1, len(self.samples)):
            indices = tuple(range(end - self.sequence_length + 1, end + 1))
            rows = [self.samples[index] for index in indices]
            if len({row.get("run_id") for row in rows}) != 1:
                continue
            if require_contiguous_frames:
                frames = [row.get("frame") for row in rows]
                if any(not isinstance(frame, (int, float)) for frame in frames):
                    continue
                numeric_frames = [float(cast(float, frame)) for frame in frames]
                deltas = [b - a for a, b in zip(numeric_frames, numeric_frames[1:])]
                if any(delta <= 0 for delta in deltas) or any(
                    abs(delta - deltas[0]) > 1e-6 for delta in deltas[1:]
                ):
                    continue
            windows.append(indices)
        return windows

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        indices = self.window_indices[index]
        items = [self.base[item_index] for item_index in indices]
        target = items[-1]["control"]
        metadata = dict(items[-1]["metadata"])
        frames = [item["metadata"].get("frame") for item in items]
        timestamps = [item["metadata"].get("timestamp") for item in items]
        numeric_timestamps = [value for value in timestamps if isinstance(value, (int, float))]
        horizon = (
            float(numeric_timestamps[-1] - numeric_timestamps[0])
            if len(numeric_timestamps) == len(timestamps)
            else None
        )
        metadata.update(
            {
                "sequence_length": self.sequence_length,
                "sequence_frames": frames,
                "sequence_timestamps": timestamps,
                "temporal_horizon_seconds": horizon,
            }
        )
        hard_control = abs(float(target[1])) >= 0.15
        return {
            "rgb": torch.stack([item["rgb"] for item in items], dim=0),
            "control": target,
            "bc_target": torch.stack([target[1], target[0]]),
            "sample_weight": torch.tensor(2.0 if hard_control else 1.0),
            "hard_control": hard_control,
            "metadata": metadata,
        }


def collate_bc_sequences(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate temporal windows while retaining metadata for diagnostics."""

    return {
        "rgb": torch.stack([item["rgb"] for item in batch], dim=0),
        "control": torch.stack([item["control"] for item in batch], dim=0),
        "bc_target": torch.stack([item["bc_target"] for item in batch], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
        "hard_control": torch.tensor([bool(item["hard_control"]) for item in batch]),
        "metadata": [dict(item["metadata"]) for item in batch],
    }
