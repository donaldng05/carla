import json
from pathlib import Path

import numpy as np

from src.perception.modality_verification import (
    discover_calibration_fields,
    estimate_front_camera_intrinsics,
    load_manual_calibration_config,
    resolve_calibration_status,
    resolve_front_camera_intrinsics,
    resolve_lidar_to_camera_extrinsics,
    resolve_projection_inputs,
    select_representative_samples,
    summarize_depth_map,
    summarize_sample,
    take_stream_window,
)


def test_estimate_front_camera_intrinsics_uses_image_center() -> None:
    k = estimate_front_camera_intrinsics(800, 600, 90.0)

    expected_focal = 400.0
    np.testing.assert_allclose(k[0, 0], expected_focal, atol=1e-6)
    np.testing.assert_allclose(k[1, 1], expected_focal, atol=1e-6)
    np.testing.assert_allclose(k[0, 2], 400.0, atol=1e-6)
    np.testing.assert_allclose(k[1, 2], 300.0, atol=1e-6)


def test_resolve_front_camera_intrinsics_prefers_explicit_matrix() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
        "camera_intrinsics": np.eye(3),
    }

    k = resolve_front_camera_intrinsics(sample)

    np.testing.assert_allclose(k, np.eye(3))


def test_resolve_lidar_to_camera_extrinsics_reads_known_key() -> None:
    sample = {
        "lidar_to_camera_extrinsics": np.eye(4),
    }

    transform = resolve_lidar_to_camera_extrinsics(sample)

    np.testing.assert_allclose(transform, np.eye(4))


def test_discover_calibration_fields_finds_matching_keys() -> None:
    sample = {
        "camera_intrinsics": np.eye(3),
        "metadata": {"lidar_pose": [0.0, 0.0, 0.0]},
        "run_id": "run-01",
    }

    discovered = discover_calibration_fields(sample)

    assert "camera_intrinsics" in discovered
    assert "metadata.lidar_pose" in discovered
    assert "run_id" not in discovered


def test_resolve_projection_inputs_reports_missing_extrinsics() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
    }

    resolved = resolve_projection_inputs(sample)

    assert resolved["can_project"] is False
    assert resolved["extrinsics"] is None
    assert resolved["extrinsics_source"] == "missing"
    assert resolved["blocking_reason"] is not None
    assert resolved["intrinsics_source"].startswith("dataset-derived")


def test_resolve_calibration_status_reports_projection_blocked() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
    }

    resolved = resolve_calibration_status(sample)

    assert resolved["can_project"] is False
    assert resolved["intrinsics_available"] is True
    assert resolved["extrinsics_available"] is False
    assert "projection blocked" in resolved["status"]
    assert resolved["fields_checked"]["extrinsics"]


def test_resolve_projection_inputs_uses_manual_override_when_sample_missing() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
    }
    manual_intrinsics = np.eye(3)
    manual_extrinsics = np.eye(4)

    resolved = resolve_projection_inputs(
        sample,
        manual_intrinsics=manual_intrinsics,
        manual_extrinsics=manual_extrinsics,
    )

    assert resolved["can_project"] is True
    np.testing.assert_allclose(resolved["intrinsics"], manual_intrinsics)
    np.testing.assert_allclose(resolved["extrinsics"], manual_extrinsics)
    assert resolved["intrinsics_source"] == "manual override"
    assert resolved["extrinsics_source"] == "manual override"


def test_resolve_projection_inputs_prefers_sample_metadata_over_manual_override() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
        "camera_intrinsics": np.eye(3) * 2.0,
        "lidar_to_camera_extrinsics": np.eye(4) * 3.0,
    }
    resolved = resolve_projection_inputs(
        sample,
        manual_intrinsics=np.eye(3),
        manual_extrinsics=np.eye(4),
    )

    np.testing.assert_allclose(resolved["intrinsics"], np.eye(3) * 2.0)
    np.testing.assert_allclose(resolved["extrinsics"], np.eye(4) * 3.0)
    assert resolved["intrinsics_source"] == "sample metadata"
    assert resolved["extrinsics_source"] == "sample metadata"


def test_load_manual_calibration_config_reads_documented_matrices(tmp_path: Path) -> None:
    config_path = tmp_path / "calibration.json"
    config_path.write_text(
        json.dumps(
            {
                "provenance": "CARLA rig notes",
                "extrinsics_source_label": "assumed rig (CARLA rig notes)",
                "camera_intrinsics": np.eye(3).tolist(),
                "lidar_to_camera_extrinsics": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )

    calibration = load_manual_calibration_config(config_path)

    assert calibration is not None
    assert "CARLA rig notes" in calibration["provenance"]
    assert calibration["extrinsics_source_label"] == "assumed rig (CARLA rig notes)"
    np.testing.assert_allclose(calibration["intrinsics"], np.eye(3))
    np.testing.assert_allclose(calibration["extrinsics"], np.eye(4))


def test_resolve_projection_inputs_uses_manual_calibration_config() -> None:
    sample = {
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
    }
    manual_calibration = {
        "intrinsics": np.eye(3),
        "extrinsics": np.eye(4),
        "provenance": "documented CARLA rig",
        "intrinsics_source_label": "dataset-derived (dataset card front camera spec)",
        "extrinsics_source_label": "assumed rig (documented CARLA rig)",
    }

    resolved = resolve_projection_inputs(sample, manual_calibration=manual_calibration)

    assert resolved["can_project"] is True
    assert resolved["intrinsics_source"] == "dataset-derived (dataset card front camera spec)"
    assert resolved["extrinsics_source"] == "assumed rig (documented CARLA rig)"


def test_repo_provisional_rig_config_loads_with_assumed_rig_label() -> None:
    calibration = load_manual_calibration_config("configs/phase1_provisional_carla_rig.json")

    assert calibration is not None
    assert calibration["extrinsics_source_label"].startswith("assumed rig")
    assert "intrinsics" not in calibration
    np.testing.assert_equal(calibration["extrinsics"].shape, (4, 4))


def test_summarize_sample_reports_modalities_and_shape() -> None:
    sample = {
        "run_id": "run-01",
        "frame": 12,
        "timestamp": 1.5,
        "image_front": np.zeros((600, 800, 3), dtype=np.uint8),
        "seg_front": np.zeros((600, 800), dtype=np.uint8),
        "lidar": np.zeros((128, 4), dtype=np.float32),
        "boxes": [[0.0, 1.0, 2.0, 3.0]],
    }

    summary = summarize_sample(sample)

    assert summary["has_image"] is True
    assert summary["has_lidar"] is True
    assert summary["has_segmentation"] is True
    assert summary["image_width"] == 800
    assert summary["image_height"] == 600
    assert summary["lidar_points"] == 128


def test_summarize_depth_map_reports_coverage() -> None:
    depth = np.array([[1.0, np.nan], [3.0, 4.0]])

    summary = summarize_depth_map(depth)

    np.testing.assert_allclose(summary["coverage"], 0.75)
    np.testing.assert_allclose(summary["min_depth"], 1.0)
    np.testing.assert_allclose(summary["max_depth"], 4.0)


def test_select_representative_samples_spreads_across_runs() -> None:
    dataset = [
        {"run_id": "a", "frame": 0},
        {"run_id": "a", "frame": 1},
        {"run_id": "b", "frame": 0},
        {"run_id": "b", "frame": 1},
        {"run_id": "c", "frame": 0},
    ]

    selected = select_representative_samples(dataset, target_count=4, per_run=1)

    assert [sample["run_id"] for sample in selected] == ["a", "b", "c"]


def test_select_representative_samples_accepts_iterators() -> None:
    dataset = iter(
        [
            {"run_id": "a", "frame": 0},
            {"run_id": "a", "frame": 1},
            {"run_id": "b", "frame": 0},
            {"run_id": "c", "frame": 0},
            {"run_id": "c", "frame": 1},
        ]
    )

    selected = select_representative_samples(dataset, target_count=3, per_run=1)

    assert [sample["run_id"] for sample in selected] == ["a", "b", "c"]


def test_take_stream_window_stops_at_target_count() -> None:
    dataset = ({"run_id": "a", "frame": idx} for idx in range(10))

    selected = take_stream_window(dataset, target_count=4)

    assert [sample["frame"] for sample in selected] == [0, 1, 2, 3]
