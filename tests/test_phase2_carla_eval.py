from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from src.evaluation.phase2_carla_eval import (
    Phase2CarlaEvalConfig,
    ego_pose_from_sample,
    iter_shadow_records,
    run_bounded_carla_iou_evaluation,
)
from src.perception.occupancy_grid import OccupancyGridSpec
from src.transforms import transform_points


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
    assert baseline["classes"][0]["class_name"] == "road"
    assert temporal["metric"] == "semantic_occupancy_iou"
    assert len(list(iter_shadow_records(artifacts.shadow_records))) == 2
