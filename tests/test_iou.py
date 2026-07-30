import json
from pathlib import Path

import numpy as np

from src.evaluation.iou import (
    compute_class_iou,
    compute_semantic_iou,
    summarize_iou,
    write_iou_summary_json,
)
from src.perception.occupancy_grid import OccupancyGridSpec, SemanticOccupancyGrid


def grid(labels: np.ndarray) -> SemanticOccupancyGrid:
    spec = OccupancyGridSpec(voxel_size_m=1.0, x_range_m=(0, 2), y_range_m=(0, 2), z_range_m=(0, 1))
    occupied = labels != -1
    counts = occupied.astype(np.uint16)
    return SemanticOccupancyGrid(spec, occupied, labels.astype(np.int16), counts)


def test_compute_class_iou_counts_intersection_and_union() -> None:
    predicted = np.array([[1, 1], [-1, 2]])
    target = np.array([[1, 2], [1, 2]])

    result = compute_class_iou(predicted, target, class_id=1)

    assert result.intersection == 1
    assert result.union == 3
    assert result.iou == 1 / 3


def test_compute_semantic_iou_accepts_occupancy_grids() -> None:
    predicted = grid(np.array([[[1], [2]], [[-1], [2]]]))
    target = grid(np.array([[[1], [1]], [[-1], [2]]]))

    results = compute_semantic_iou(predicted, target, class_ids=(1, 2))

    assert results[1].iou == 0.5
    assert results[2].iou == 0.5


def test_summarize_iou_aggregates_intersections_and_unions() -> None:
    first = {1: compute_class_iou(np.array([1, 2]), np.array([1, 1]), class_id=1)}
    second = {1: compute_class_iou(np.array([1, 1]), np.array([2, 1]), class_id=1)}

    summary = summarize_iou([first, second])

    assert summary[1].intersection == 2
    assert summary[1].union == 4
    assert summary[1].iou == 0.5


def test_write_iou_summary_json_creates_readable_artifact(tmp_path: Path) -> None:
    results = {1: compute_class_iou(np.array([1, 2]), np.array([1, 1]), class_id=1)}

    path = write_iou_summary_json(results, tmp_path / "iou_summary.json", class_names={1: "road"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metric"] == "semantic_occupancy_iou"
    assert payload["classes"][0]["class_name"] == "road"
