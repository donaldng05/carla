from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data import (
    CarlaDataPreprocessingConfig,
    CarlaMultimodalDataset,
    audit_split_run_ids,
    build_carla_dataset_from_hf,
    collate_carla_samples,
    validate_processed_dataset,
)
from src.data.carla_dataset import DEFAULT_CALIBRATION_CONFIG_PATH


def _make_sample(*, segmentation_value: int = 1) -> dict[str, Any]:
    return {
        "run_id": "run-01",
        "frame": 3,
        "timestamp": 1.5,
        "speed_kmh": 24.0,
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
        "throttle": 0.4,
        "steer": -0.1,
        "brake": 0.0,
        "weather_cloudiness": 10.0,
    }


def test_dataset_returns_aligned_multimodal_sample() -> None:
    dataset = CarlaMultimodalDataset([_make_sample()])

    item = dataset[0]

    assert set(item) == {"rgb", "depth", "segmentation", "control", "lidar", "metadata"}
    assert item["rgb"].shape == (3, 6, 8)
    assert item["depth"].shape == (1, 6, 8)
    assert item["segmentation"].shape == (6, 8)
    assert item["control"].shape == (3,)
    assert item["lidar"].shape == (2, 4)
    assert item["metadata"]["run_id"] == "run-01"
    assert item["metadata"]["depth_summary"]["coverage"] > 0.0


def test_default_calibration_config_path_is_repo_absolute() -> None:
    assert DEFAULT_CALIBRATION_CONFIG_PATH.is_absolute()
    assert DEFAULT_CALIBRATION_CONFIG_PATH.name == "phase1_provisional_carla_rig.json"
    assert DEFAULT_CALIBRATION_CONFIG_PATH.exists()


def test_dataset_hole_fill_is_optional_postprocess() -> None:
    sample = _make_sample()
    sample["lidar"] = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.625, 0.0, 0.0],
            [4.0, 1.25, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    base_dataset = CarlaMultimodalDataset([sample])
    filled_dataset = CarlaMultimodalDataset(
        [sample],
        preprocessing=CarlaDataPreprocessingConfig(hole_fill=True),
    )

    base_depth = base_dataset[0]["depth"]
    filled_depth = filled_dataset[0]["depth"]

    assert torch.isnan(base_depth).any()
    assert torch.isfinite(filled_depth).sum() >= torch.isfinite(base_depth).sum()


def test_dataset_applies_segmentation_remap() -> None:
    config = CarlaDataPreprocessingConfig(segmentation_remap={7: 3})
    dataset = CarlaMultimodalDataset([_make_sample(segmentation_value=7)], preprocessing=config)

    item = dataset[0]

    assert torch.unique(item["segmentation"]).tolist() == [3]


def test_dataset_decodes_colorized_carla_segmentation() -> None:
    sample = _make_sample()
    sample["seg_front"] = np.zeros((6, 8, 4), dtype=np.uint8)
    sample["seg_front"][..., :3] = np.array([128, 64, 128], dtype=np.uint8)
    sample["seg_front"][..., 3] = 255
    sample["seg_front"][0, 0, :3] = np.array([0, 0, 142], dtype=np.uint8)

    dataset = CarlaMultimodalDataset([sample])

    item = dataset[0]

    assert torch.unique(item["segmentation"]).tolist() == [7, 10]


def test_dataset_maps_cityscapes_vehicle_family_colors_to_vehicle_class() -> None:
    sample = _make_sample()
    sample["seg_front"] = np.zeros((6, 8, 4), dtype=np.uint8)
    sample["seg_front"][..., :3] = np.array([0, 60, 100], dtype=np.uint8)
    sample["seg_front"][..., 3] = 255
    sample["seg_front"][0, 0, :3] = np.array([0, 0, 230], dtype=np.uint8)
    sample["seg_front"][0, 1, :3] = np.array([119, 11, 32], dtype=np.uint8)

    dataset = CarlaMultimodalDataset([sample])

    item = dataset[0]

    assert torch.unique(item["segmentation"]).tolist() == [10]


def test_preprocessing_resizes_and_normalizes() -> None:
    config = CarlaDataPreprocessingConfig(
        target_image_size=(4, 4),
        normalize_rgb=True,
        hole_fill=True,
    )
    dataset = CarlaMultimodalDataset([_make_sample()], preprocessing=config)

    item = dataset[0]

    assert item["rgb"].shape == (3, 4, 4)
    assert item["depth"].shape == (1, 4, 4)
    assert item["segmentation"].shape == (4, 4)
    assert item["rgb"].dtype == torch.float32
    assert torch.max(item["rgb"]) < 1.0


def test_collate_carla_samples_stacks_fixed_modalities_and_keeps_lidar_list() -> None:
    dataset = CarlaMultimodalDataset([_make_sample(), _make_sample(segmentation_value=2)])
    batch = collate_carla_samples([dataset[0], dataset[1]])

    assert batch["rgb"].shape == (2, 3, 6, 8)
    assert batch["depth"].shape == (2, 1, 6, 8)
    assert batch["segmentation"].shape == (2, 6, 8)
    assert batch["control"].shape == (2, 3)
    assert len(batch["lidar"]) == 2
    assert len(batch["metadata"]) == 2


def test_validate_processed_dataset_reports_malformed_samples() -> None:
    broken = _make_sample()
    broken.pop("seg_front")
    dataset = CarlaMultimodalDataset([_make_sample(), broken])

    summary = validate_processed_dataset(dataset)

    assert summary["total_samples"] == 2
    assert summary["success_count"] == 1
    assert summary["malformed_count"] == 1
    assert summary["error_counts"]["KeyError"] == 1


def test_audit_split_run_ids_passes_for_disjoint_runs() -> None:
    report = audit_split_run_ids(
        {
            "train": [{"run_id": "run-01"}, {"run_id": "run-02"}],
            "validation": [{"run_id": "run-03"}],
            "test": [{"run_id": "run-04"}],
        }
    )

    assert report["has_overlap"] is False
    assert report["overlaps"] == {}


def test_audit_split_run_ids_reports_overlap() -> None:
    report = audit_split_run_ids(
        {
            "train": [{"run_id": "run-01"}],
            "validation": [{"run_id": "run-01"}],
            "test": [{"run_id": "run-02"}],
        }
    )

    assert report["has_overlap"] is True
    assert report["overlaps"]["run-01"] == ["train", "validation"]


def test_build_carla_dataset_from_hf_wraps_loaded_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | Path | None, bool]] = []

    def fake_load_hf_split(
        dataset_name: str,
        split: str,
        *,
        cache_dir: str | Path | None = None,
        streaming: bool = False,
    ) -> list[dict[str, Any]]:
        calls.append((dataset_name, split, cache_dir, streaming))
        return [_make_sample()]

    monkeypatch.setattr("src.data.carla_dataset._load_hf_split", fake_load_hf_split)

    dataset = build_carla_dataset_from_hf(split="validation")

    assert isinstance(dataset, CarlaMultimodalDataset)
    assert len(dataset) == 1
    assert calls == [
        ("immanuelpeter/carla-autopilot-multimodal-dataset", "validation", None, False)
    ]
