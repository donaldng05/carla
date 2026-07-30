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
    ScenarioMemoryEvalConfig,
    _anchor_indices,
    ego_pose_from_sample,
    iter_shadow_records,
    run_bounded_carla_iou_evaluation,
    run_future_pseudo_label_evaluation,
    run_scenario_memory_evaluation,
    score_scenario_memory_candidates,
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


def _multi_voxel_grid(
    spec: OccupancyGridSpec,
    voxels: list[tuple[tuple[int, int, int], int]],
) -> SemanticOccupancyGrid:
    occupied = np.zeros(spec.shape, dtype=bool)
    semantic = np.full(spec.shape, UNKNOWN_CLASS, dtype=np.int16)
    counts = np.zeros(spec.shape, dtype=np.uint16)
    for voxel, label in voxels:
        occupied[voxel] = True
        semantic[voxel] = label
        counts[voxel] = 1
    return SemanticOccupancyGrid(spec, occupied, semantic, counts)


def _occupancy_frame(
    spec: OccupancyGridSpec,
    *,
    frame: int,
    voxels: list[tuple[tuple[int, int, int], int]],
    coverage: float,
    steer: float = 0.0,
    yaw: float = 0.0,
    nearby: int = 0,
    total_vehicles: int = 0,
) -> OccupancyFrame:
    return OccupancyFrame(
        _multi_voxel_grid(spec, voxels),
        carla_pose_to_matrix(x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0),
        metadata={
            "run_id": "scenario-test",
            "frame": frame,
            "steer": steer,
            "rotation_yaw": yaw,
            "nearby_vehicles_50m": nearby,
            "total_npc_vehicles": total_vehicles,
            "depth_summary": {"coverage": coverage},
        },
    )


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


def _scenario_frames() -> tuple[OccupancyGridSpec, list[OccupancyFrame]]:
    spec = OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(0.0, 4.0),
        y_range_m=(0.0, 4.0),
        z_range_m=(0.0, 2.0),
    )
    frames = [
        _occupancy_frame(
            spec,
            frame=0,
            voxels=[((0, 0, 0), 10)],
            coverage=0.9,
            yaw=0.0,
        ),
        _occupancy_frame(
            spec,
            frame=1,
            voxels=[((1, 1, 0), 7)],
            coverage=0.01,
            steer=0.2,
            yaw=8.0,
            nearby=12,
            total_vehicles=12,
        ),
        _occupancy_frame(
            spec,
            frame=2,
            voxels=[((1, 1, 0), 7), ((2, 1, 0), 10)],
            coverage=0.8,
            yaw=8.0,
        ),
        _occupancy_frame(
            spec,
            frame=3,
            voxels=[((1, 2, 0), 7)],
            coverage=0.7,
            yaw=8.0,
        ),
    ]
    return spec, frames


def test_scenario_memory_selector_catches_all_synthetic_conditions() -> None:
    spec, frames = _scenario_frames()
    config = ScenarioMemoryEvalConfig(
        max_scan_frames=4,
        anchors_per_slice=2,
        past_window=2,
        future_window=2,
        grid_spec=spec,
        class_ids=(7, 10, 4),
        min_vehicle_union=1,
        occupancy_threshold=0.25,
        turn_steer_threshold=0.15,
        dense_traffic_min_nearby=10,
        disagreement_threshold=0.01,
    )

    candidates, summary = score_scenario_memory_candidates(frames, config=config)

    assert summary["candidate_count"] == 2
    anchor_one = next(candidate.anchor for candidate in candidates if candidate.anchor.frame == 1)
    assert set(anchor_one.slice_names) == {
        "vehicle_change",
        "sparse_depth",
        "turning",
        "dense_traffic",
        "high_disagreement",
    }
    assert anchor_one.new_vehicle_voxel_count >= 1


def test_scenario_memory_evaluation_writes_jsonl_and_per_slice_iou(tmp_path: Path) -> None:
    spec, frames = _scenario_frames()
    config = ScenarioMemoryEvalConfig(
        max_scan_frames=4,
        anchors_per_slice=1,
        output_dir=tmp_path,
        past_window=2,
        future_window=2,
        grid_spec=spec,
        class_ids=(7, 10, 4),
        min_vehicle_union=1,
        occupancy_threshold=0.25,
        disagreement_threshold=0.01,
        bev_frame_count_per_slice=0,
    )

    artifacts = run_scenario_memory_evaluation(config=config, frames=frames)

    assert artifacts.eval_summary.exists()
    assert artifacts.by_slice_iou.exists()
    assert artifacts.selected_anchors.exists()
    assert artifacts.bev_images == ()

    records = [
        json.loads(line)
        for line in artifacts.selected_anchors.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {record["slice"] for record in records} >= {
        "vehicle_change",
        "sparse_depth",
        "turning",
        "dense_traffic",
        "high_disagreement",
    }

    by_slice = json.loads(artifacts.by_slice_iou.read_text(encoding="utf-8"))
    vehicle_rows = {
        row["class_name"]: row for row in by_slice["slices"]["vehicle_change"]["classes"]
    }
    assert set(vehicle_rows) == {"road", "vehicle", "pedestrian"}
    assert vehicle_rows["vehicle"]["informative"] is True
    assert vehicle_rows["pedestrian"]["informative"] is False

    summary = json.loads(artifacts.eval_summary.read_text(encoding="utf-8"))
    assert summary["selected_anchor_counts"]["vehicle_change"] == 1
    assert summary["selection_thresholds"]["dense_traffic_min_nearby"] == 10
