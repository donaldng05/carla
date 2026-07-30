from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from src.evaluation.phase2_carla_eval import (
    FuturePseudoEvalConfig,
    Phase2CarlaEvalConfig,
    _anchor_indices,
    ego_pose_from_sample,
    iter_shadow_records,
    run_bounded_carla_iou_evaluation,
    run_future_pseudo_label_evaluation,
    union_grids_in_ego_frame,
)
from src.perception.occupancy_grid import UNKNOWN_CLASS, OccupancyGridSpec, SemanticOccupancyGrid
from src.perception.temporal_fusion import OccupancyFrame
from src.transforms import carla_pose_to_matrix, transform_points


def _sample(*, frame: int, segmentation_value: int = 7, location_x: float = 0.0) -> dict[str, Any]:
    return {
        "run_id": "test-run",
        "frame": frame,
        "timestamp": float(frame) * 0.1,
        "speed_kmh": 20.0 + frame,
        "map_name": "Town01",
        "image_front": np.full((6, 8, 3), 64, dtype=np.uint8),
        "seg_front": np.full((6, 8), segmentation_value, dtype=np.uint8),
        "lidar": np.array(
            [
                [4.0, 0.0, 0.0, 0.8],
                [4.0, 0.625, 0.0, 0.5],
            ],
            dtype=np.float32,
        ),
        "location_x": location_x,
        "location_y": 0.0,
        "location_z": 0.0,
        "rotation_pitch": 0.0,
        "rotation_yaw": 0.0,
        "rotation_roll": 0.0,
        "throttle": 0.4,
        "steer": -0.1,
        "brake": 0.0,
        "weather_precipitation": 0.0,
        "weather_fog_density": 0.0,
        "weather_sun_altitude_angle": 45.0,
        "nearby_vehicles_50m": 3,
    }


def test_ego_pose_from_sample_uses_location_fields() -> None:
    pose = ego_pose_from_sample(_sample(frame=0, location_x=3.0))

    transformed = transform_points(np.array([1.0, 0.0, 0.0]), pose)

    np.testing.assert_allclose(transformed, np.array([4.0, 0.0, 0.0]))


def test_run_bounded_carla_iou_evaluation_writes_artifacts(tmp_path: Path) -> None:
    config = Phase2CarlaEvalConfig(
        max_frames=2,
        output_dir=tmp_path,
        grid_spec=OccupancyGridSpec(
            voxel_size_m=1.0,
            x_range_m=(-2.0, 6.0),
            y_range_m=(-3.0, 3.0),
            z_range_m=(0.0, 6.0),
        ),
        class_ids=(7, 10, 4),
        bev_frame_count=1,
    )

    artifacts = run_bounded_carla_iou_evaluation(
        config=config,
        raw_samples=[_sample(frame=0), _sample(frame=1, segmentation_value=10)],
    )

    assert artifacts.baseline_iou.exists()
    assert artifacts.temporal_iou.exists()
    assert artifacts.shadow_records.exists()
    assert artifacts.shadow_clusters.exists()
    assert artifacts.run_summary.exists()
    assert len(artifacts.bev_images) == 1
    assert artifacts.bev_images[0].exists()

    summary = json.loads(artifacts.run_summary.read_text(encoding="utf-8"))
    assert summary["processed_frame_count"] == 2
    assert summary["target_proxy"].startswith("current-frame occupancy grid")

    baseline = json.loads(artifacts.baseline_iou.read_text(encoding="utf-8"))
    temporal = json.loads(artifacts.temporal_iou.read_text(encoding="utf-8"))
    assert {row["class_name"] for row in baseline["classes"]} == {"road", "vehicle", "pedestrian"}
    assert temporal["metric"] == "semantic_occupancy_iou"
    assert len(list(iter_shadow_records(artifacts.shadow_records))) == 2


def _single_voxel_grid(
    spec: OccupancyGridSpec,
    *,
    voxel: tuple[int, int, int],
    label: int,
) -> SemanticOccupancyGrid:
    occupied = np.zeros(spec.shape, dtype=bool)
    semantic = np.full(spec.shape, UNKNOWN_CLASS, dtype=np.int16)
    counts = np.zeros(spec.shape, dtype=np.uint16)
    occupied[voxel] = True
    semantic[voxel] = label
    counts[voxel] = 1
    return SemanticOccupancyGrid(spec, occupied, semantic, counts)


def test_future_pseudo_target_unions_future_grids_in_anchor_frame() -> None:
    spec = OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(0.0, 4.0),
        y_range_m=(0.0, 4.0),
        z_range_m=(0.0, 2.0),
    )
    anchor_pose = carla_pose_to_matrix(x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0)
    future_pose = carla_pose_to_matrix(x=1.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0)
    anchor = OccupancyFrame(
        _single_voxel_grid(spec, voxel=(1, 1, 0), label=7),
        anchor_pose,
    )
    future = OccupancyFrame(
        _single_voxel_grid(spec, voxel=(1, 1, 0), label=10),
        future_pose,
    )

    target = union_grids_in_ego_frame((anchor, future), target_ego_pose=anchor_pose, spec=spec)

    assert target.occupied[1, 1, 0]
    assert target.occupied[2, 1, 0]
    assert target.semantic[1, 1, 0] == 7
    assert target.semantic[2, 1, 0] == 10


def test_anchor_indices_skip_when_context_is_insufficient() -> None:
    assert _anchor_indices(frame_count=3, anchor_count=10, past_window=3, future_window=2) == ()
    assert _anchor_indices(frame_count=6, anchor_count=10, past_window=2, future_window=2) == (
        1,
        2,
        3,
        4,
    )


def test_future_pseudo_evaluation_writes_artifacts_and_ablation_rows(tmp_path: Path) -> None:
    config = FuturePseudoEvalConfig(
        anchor_count=2,
        past_window=2,
        future_window=2,
        output_dir=tmp_path,
        thresholds=(0.25, 0.5),
        grid_spec=OccupancyGridSpec(
            voxel_size_m=1.0,
            x_range_m=(-2.0, 6.0),
            y_range_m=(-3.0, 3.0),
            z_range_m=(0.0, 6.0),
        ),
        class_ids=(7, 10, 4),
        bev_frame_count=1,
        alignment_frame_count=1,
    )

    artifacts = run_future_pseudo_label_evaluation(
        config=config,
        raw_samples=[
            _sample(frame=0, segmentation_value=7, location_x=0.0),
            _sample(frame=1, segmentation_value=7, location_x=0.2),
            _sample(frame=2, segmentation_value=10, location_x=0.4),
            _sample(frame=3, segmentation_value=10, location_x=0.6),
        ],
    )

    assert artifacts.baseline_iou.exists()
    assert artifacts.temporal_iou.exists()
    assert artifacts.eval_summary.exists()
    assert artifacts.shadow_records.exists()
    assert artifacts.shadow_clusters.exists()
    assert artifacts.alignment_summary.exists()
    assert artifacts.ablation_summary.exists()
    assert len(artifacts.bev_images) == 1
    assert len(artifacts.alignment_images) == 1

    eval_summary = json.loads(artifacts.eval_summary.read_text(encoding="utf-8"))
    assert eval_summary["processed_anchor_count"] == 2
    assert eval_summary["target_proxy"].startswith("future union")

    ablation = json.loads(artifacts.ablation_summary.read_text(encoding="utf-8"))
    assert len(ablation["rows"]) == 4
    assert "selected_profile" in ablation

    alignment = json.loads(artifacts.alignment_summary.read_text(encoding="utf-8"))
    assert alignment["near_overlap_tolerance_voxels"] == 2
    assert "mean_previous_near_overlap_ratio" in alignment

    baseline = json.loads(artifacts.baseline_iou.read_text(encoding="utf-8"))
    assert {row["class_name"] for row in baseline["classes"]} == {"road", "vehicle", "pedestrian"}
