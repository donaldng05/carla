"""IoU metrics for semantic occupancy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src.perception.occupancy_grid import UNKNOWN_CLASS, SemanticOccupancyGrid
from src.perception.temporal_fusion import FusedOccupancyGrid


@dataclass(frozen=True)
class ClassIoU:
    """Intersection-over-union result for one semantic class."""

    class_id: int
    intersection: int
    union: int
    iou: float


def semantic_and_occupied(
    grid: SemanticOccupancyGrid | FusedOccupancyGrid,
    *,
    occupancy_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract semantic labels and binary occupancy from a supported grid type."""

    if isinstance(grid, SemanticOccupancyGrid):
        return grid.semantic, grid.occupied
    if isinstance(grid, FusedOccupancyGrid):
        return grid.semantic, grid.occupancy_score >= occupancy_threshold
    raise TypeError("grid must be a SemanticOccupancyGrid or FusedOccupancyGrid")


def compute_class_iou(
    predicted_semantic: np.ndarray,
    target_semantic: np.ndarray,
    *,
    class_id: int,
    predicted_occupied: np.ndarray | None = None,
    target_occupied: np.ndarray | None = None,
) -> ClassIoU:
    """Compute IoU for one semantic class."""

    predicted = np.asarray(predicted_semantic)
    target = np.asarray(target_semantic)
    if predicted.shape != target.shape:
        raise ValueError("predicted_semantic and target_semantic must have matching shapes")

    pred_occ = (
        predicted != UNKNOWN_CLASS
        if predicted_occupied is None
        else np.asarray(predicted_occupied, dtype=bool)
    )
    target_occ = (
        target != UNKNOWN_CLASS
        if target_occupied is None
        else np.asarray(target_occupied, dtype=bool)
    )
    if pred_occ.shape != predicted.shape or target_occ.shape != target.shape:
        raise ValueError("occupancy masks must match semantic grid shape")

    predicted_mask = pred_occ & (predicted == class_id)
    target_mask = target_occ & (target == class_id)
    intersection = int(np.count_nonzero(predicted_mask & target_mask))
    union = int(np.count_nonzero(predicted_mask | target_mask))
    iou = float(intersection / union) if union else 1.0
    return ClassIoU(class_id=int(class_id), intersection=intersection, union=union, iou=iou)


def compute_semantic_iou(
    predicted: SemanticOccupancyGrid | FusedOccupancyGrid,
    target: SemanticOccupancyGrid | FusedOccupancyGrid,
    *,
    class_ids: Iterable[int],
    occupancy_threshold: float = 0.5,
) -> dict[int, ClassIoU]:
    """Compute per-class IoU between predicted and target semantic occupancy grids."""

    predicted_semantic, predicted_occupied = semantic_and_occupied(
        predicted,
        occupancy_threshold=occupancy_threshold,
    )
    target_semantic, target_occupied = semantic_and_occupied(
        target,
        occupancy_threshold=occupancy_threshold,
    )
    if predicted_semantic.shape != target_semantic.shape:
        raise ValueError("predicted and target grids must have matching shapes")

    return {
        int(class_id): compute_class_iou(
            predicted_semantic,
            target_semantic,
            class_id=int(class_id),
            predicted_occupied=predicted_occupied,
            target_occupied=target_occupied,
        )
        for class_id in class_ids
    }


def summarize_iou(results: Iterable[Mapping[int, ClassIoU]]) -> dict[int, ClassIoU]:
    """Aggregate per-frame IoU results by summing intersections and unions."""

    totals: dict[int, tuple[int, int]] = {}
    for frame_result in results:
        for class_id, result in frame_result.items():
            intersection, union = totals.get(int(class_id), (0, 0))
            totals[int(class_id)] = (
                intersection + int(result.intersection),
                union + int(result.union),
            )

    return {
        class_id: ClassIoU(
            class_id=class_id,
            intersection=intersection,
            union=union,
            iou=float(intersection / union) if union else 1.0,
        )
        for class_id, (intersection, union) in totals.items()
    }


def iou_table(
    results: Mapping[int, ClassIoU], *, class_names: Mapping[int, str] | None = None
) -> list[dict[str, float | int | str]]:
    """Convert IoU results into table-ready dictionaries."""

    names = class_names or {}
    return [
        {
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "intersection": result.intersection,
            "union": result.union,
            "iou": result.iou,
        }
        for class_id, result in sorted(results.items())
    ]


def write_iou_summary_json(
    results: Mapping[int, ClassIoU],
    output_path: str | Path,
    *,
    class_names: Mapping[int, str] | None = None,
) -> Path:
    """Write aggregate IoU results as a readable JSON artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "semantic_occupancy_iou",
        "classes": iou_table(results, class_names=class_names),
        "interpretation": (
            "IoU = intersection / union for occupied voxels of each semantic class. "
            "Higher is better; 1.0 is perfect overlap. Empty-union classes are reported as 1.0."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
