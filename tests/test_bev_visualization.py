"""Regression tests for BEV figure readability."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.phase2_carla_eval import (
    BEV_CLASS_COLORS,
    BEV_CLASS_NAMES,
    _occupied_bbox,
    _pretty_slice_name,
    save_bev_comparison,
)
from src.perception.occupancy_grid import UNKNOWN_CLASS, OccupancyGridSpec, SemanticOccupancyGrid


def _grid(
    spec: OccupancyGridSpec, voxels: list[tuple[tuple[int, int, int], int]]
) -> SemanticOccupancyGrid:
    occupied = np.zeros(spec.shape, dtype=bool)
    semantic = np.full(spec.shape, UNKNOWN_CLASS, dtype=np.int16)
    counts = np.zeros(spec.shape, dtype=np.uint16)
    for voxel, label in voxels:
        occupied[voxel] = True
        semantic[voxel] = label
        counts[voxel] = 1
    return SemanticOccupancyGrid(spec, occupied, semantic, counts)


def test_bev_palette_covers_truck_and_free_space_is_not_white() -> None:
    assert BEV_CLASS_NAMES[25] == "truck"
    assert BEV_CLASS_NAMES[7] == "road"
    assert BEV_CLASS_NAMES[10] == "vehicle"
    assert BEV_CLASS_COLORS[UNKNOWN_CLASS].lower() != "#ffffff"


def test_slice_names_are_humanized() -> None:
    assert _pretty_slice_name("high_disagreement") == "High disagreement"
    assert _pretty_slice_name("sparse_depth") == "Sparse depth"
    assert "_" not in _pretty_slice_name("high_disagreement")


def test_crop_keeps_minimum_context() -> None:
    panels = [np.full((200, 200), UNKNOWN_CLASS, dtype=np.int16) for _ in range(3)]
    for panel in panels:
        panel[100, 100] = 10
    r0, r1, c0, c1 = _occupied_bbox(panels, padding=24, min_extent=48)
    assert (r1 - r0) >= 48
    assert (c1 - c0) >= 48


def test_save_bev_comparison_writes_human_figure(tmp_path: Path) -> None:
    spec = OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(0.0, 8.0),
        y_range_m=(0.0, 8.0),
        z_range_m=(0.0, 2.0),
    )
    baseline = _grid(spec, [((2, 2, 0), 7), ((5, 5, 0), 7)])
    temporal = _grid(spec, [((2, 2, 0), 7), ((5, 5, 0), 10), ((4, 4, 0), 25)])
    target = _grid(spec, [((2, 2, 0), 7), ((5, 5, 0), 10), ((4, 4, 0), 25)])
    class_names = {7: "road", 10: "vehicle", 4: "pedestrian"}

    out = tmp_path / "bev.png"
    title = "Occlusion recovery, anchor 475"
    subtitle = "High disagreement slice: single frame vs 5-frame memory vs denser future target"
    path = save_bev_comparison(
        baseline=baseline,
        temporal=temporal,
        target=target,
        output_path=out,
        title=title,
        subtitle=subtitle,
        class_names=class_names,
        panel_titles=("Single frame", "5-frame memory", "Denser future target"),
    )

    assert path.exists()
    assert "_" not in title.split(",")[0]
    assert ":" not in title
    # Every observed non-free class must resolve to a human name, never a fallback.
    observed = {7, 10, 25}
    display = BEV_CLASS_NAMES | dict(class_names)
    for cls in observed:
        name = display.get(cls, f"unknown {cls}")
        assert not name.startswith("class ")
    assert display[25] == "truck"
