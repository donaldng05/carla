"""Temporal fusion for semantic occupancy grids."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.perception.occupancy_grid import (
    UNKNOWN_CLASS,
    OccupancyGridSpec,
    SemanticOccupancyGrid,
    voxel_indices_to_points,
    voxelize_points,
)
from src.transforms import vehicle_to_world, world_to_vehicle

DEFAULT_FUSION_WEIGHTS = (1.0, 0.8, 0.6, 0.4, 0.2)


@dataclass(frozen=True)
class OccupancyFrame:
    """One timestamped occupancy grid in its own ego-vehicle frame."""

    grid: SemanticOccupancyGrid
    ego_pose: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FusedOccupancyGrid:
    """Weighted temporal occupancy scores aligned to the current ego frame."""

    spec: OccupancyGridSpec
    occupancy_score: np.ndarray
    semantic: np.ndarray
    weights: tuple[float, ...]

    @property
    def occupied(self) -> np.ndarray:
        return self.occupancy_score > 0.0


def _validate_pose(pose: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    return matrix


def transform_grid_to_ego_frame(
    grid: SemanticOccupancyGrid,
    *,
    source_ego_pose: np.ndarray | list[list[float]],
    target_ego_pose: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform occupied voxel centers from a source ego frame into a target ego frame."""

    source_pose = _validate_pose(source_ego_pose, "source_ego_pose")
    target_pose = _validate_pose(target_ego_pose, "target_ego_pose")
    source_indices = np.argwhere(grid.occupied)
    if source_indices.size == 0:
        empty_indices = np.empty((0, 3), dtype=np.int64)
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_counts = np.empty((0,), dtype=np.float64)
        return empty_indices, empty_labels, empty_counts

    source_points = voxel_indices_to_points(source_indices, grid.spec)
    world_points = vehicle_to_world(source_points, source_pose)
    target_points = world_to_vehicle(world_points, target_pose)
    target_indices, valid = voxelize_points(target_points, grid.spec)
    if target_indices.size == 0:
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_counts = np.empty((0,), dtype=np.float64)
        return target_indices, empty_labels, empty_counts

    source_indices = source_indices[valid]
    labels = grid.semantic[source_indices[:, 0], source_indices[:, 1], source_indices[:, 2]]
    counts = grid.counts[source_indices[:, 0], source_indices[:, 1], source_indices[:, 2]].astype(
        np.float64
    )
    return target_indices, labels.astype(np.int64), counts


def fuse_occupancy_frames(
    frames_newest_first: Sequence[OccupancyFrame],
    *,
    weights: Iterable[float] = DEFAULT_FUSION_WEIGHTS,
    target_ego_pose: np.ndarray | list[list[float]] | None = None,
) -> FusedOccupancyGrid:
    """Fuse occupancy frames into a target ego frame using weighted binary occupancy evidence."""

    frames = list(frames_newest_first)
    if not frames:
        raise ValueError("frames_newest_first must contain at least one frame")

    weight_values = tuple(float(weight) for weight in weights)
    if not weight_values:
        raise ValueError("weights must contain at least one value")
    if any(weight < 0.0 for weight in weight_values):
        raise ValueError("weights must be non-negative")

    current = frames[0]
    spec = current.grid.spec
    shape = spec.shape
    score_sum = np.zeros(shape, dtype=np.float64)
    semantic = np.full(shape, UNKNOWN_CLASS, dtype=np.int16)
    semantic_votes: dict[tuple[int, int], float] = {}

    used_weights = weight_values[: len(frames)]
    total_weight = sum(used_weights)
    if total_weight <= 0.0:
        raise ValueError("at least one fusion weight must be positive")

    target_pose = (
        current.ego_pose
        if target_ego_pose is None
        else _validate_pose(target_ego_pose, "target_ego_pose")
    )

    for frame, weight in zip(frames, used_weights):
        if frame.grid.spec != spec:
            raise ValueError("all occupancy frames must use the same grid spec")
        if weight == 0.0:
            continue

        indices, labels, counts = transform_grid_to_ego_frame(
            frame.grid,
            source_ego_pose=frame.ego_pose,
            target_ego_pose=target_pose,
        )
        if indices.size == 0:
            continue

        linear = np.ravel_multi_index((indices[:, 0], indices[:, 1], indices[:, 2]), shape)
        unique_linear = np.unique(linear)
        score_sum.flat[unique_linear] += weight

        for linear_index, label, count in zip(linear, labels, counts):
            semantic_votes[(int(linear_index), int(label))] = semantic_votes.get(
                (int(linear_index), int(label)),
                0.0,
            ) + weight * max(float(count), 1.0)

    occupancy_score = score_sum / total_weight
    best_vote_by_voxel: dict[int, tuple[float, int]] = {}
    for (linear_index, label), vote in semantic_votes.items():
        best_vote, best_label = best_vote_by_voxel.get(linear_index, (-1.0, UNKNOWN_CLASS))
        if vote > best_vote or (np.isclose(vote, best_vote) and label < best_label):
            best_vote_by_voxel[linear_index] = (vote, label)

    for linear_index, (_, label) in best_vote_by_voxel.items():
        semantic.flat[linear_index] = label

    return FusedOccupancyGrid(spec, occupancy_score, semantic, used_weights)


class TemporalFusionBuffer:
    """Rolling 5-frame temporal fusion buffer for semantic occupancy grids."""

    def __init__(self, *, weights: Iterable[float] = DEFAULT_FUSION_WEIGHTS) -> None:
        self.weights = tuple(float(weight) for weight in weights)
        if not self.weights:
            raise ValueError("weights must contain at least one value")
        self._frames: deque[OccupancyFrame] = deque(maxlen=len(self.weights))

    def add(self, frame: OccupancyFrame) -> FusedOccupancyGrid:
        self._frames.appendleft(frame)
        return self.fuse()

    def fuse(self) -> FusedOccupancyGrid:
        return fuse_occupancy_frames(list(self._frames), weights=self.weights)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[OccupancyFrame, ...]:
        return tuple(self._frames)


def make_temporal_alignment_figure(
    current: OccupancyFrame,
    previous: OccupancyFrame,
) -> tuple[Any, Any]:
    """Visualize current occupied voxels against a previous grid aligned to current ego frame."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise ImportError("matplotlib is required for make_temporal_alignment_figure") from exc

    current_indices = np.argwhere(current.grid.occupied)
    aligned_indices, _, _ = transform_grid_to_ego_frame(
        previous.grid,
        source_ego_pose=previous.ego_pose,
        target_ego_pose=current.ego_pose,
    )

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    if current_indices.size:
        ax.scatter(current_indices[:, 0], current_indices[:, 1], s=8, c="#c83f49", label="current")
    if aligned_indices.size:
        ax.scatter(
            aligned_indices[:, 0],
            aligned_indices[:, 1],
            s=8,
            c="#4c78a8",
            alpha=0.6,
            label="previous aligned",
        )
    ax.set_title("Temporal occupancy alignment")
    ax.set_xlabel("Vehicle x voxel")
    ax.set_ylabel("Vehicle y voxel")
    ax.legend(loc="best")
    return fig, ax
