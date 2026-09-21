"""PyTorch dataset and preprocessing helpers for CARLA multimodal samples."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.split_audit import DEFAULT_DATASET_NAME
from src.perception.lidar_conversion import fill_depth_holes, lidar_to_depth
from src.perception.modality_verification import (
    estimate_front_camera_intrinsics,
    load_manual_calibration_config,
    summarize_depth_map,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "phase1_provisional_carla_rig.json"
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
CARLA_CITYSCAPES_PALETTE: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (70, 70, 70): 1,
    (100, 40, 40): 2,
    (55, 90, 80): 3,
    (220, 20, 60): 4,
    (153, 153, 153): 5,
    (157, 234, 50): 6,
    (128, 64, 128): 7,
    (244, 35, 232): 8,
    (107, 142, 35): 9,
    (0, 0, 142): 10,
    (102, 102, 156): 11,
    (220, 220, 0): 12,
    (70, 130, 180): 13,
    (81, 0, 81): 14,
    (150, 100, 100): 15,
    (230, 150, 140): 16,
    (180, 165, 180): 17,
    (250, 170, 30): 18,
    (110, 190, 160): 19,
    (170, 120, 50): 20,
    (45, 60, 150): 21,
    (145, 170, 100): 22,
    (152, 251, 152): 23,
    (190, 153, 153): 24,
    (0, 0, 70): 25,
    # The streamed dataset uses Cityscapes-style colors for vehicle-family actors
    # that are not distinct classes in the Phase 2 canonical occupancy taxonomy.
    (0, 60, 100): 10,  # bus -> vehicle
    (0, 0, 230): 10,  # motorcycle -> vehicle
    (119, 11, 32): 10,  # bicycle -> vehicle
}


@dataclass(frozen=True)
class CarlaDataPreprocessingConfig:
    """Configurable preprocessing profile for one CARLA multimodal dataset view."""

    target_image_size: tuple[int, int] | None = None
    normalize_rgb: bool = False
    rgb_mean: tuple[float, float, float] = IMAGENET_RGB_MEAN
    rgb_std: tuple[float, float, float] = IMAGENET_RGB_STD
    hole_fill: bool = False
    hole_fill_iterations: int = 1
    hole_fill_window_size: int = 3
    segmentation_remap: Mapping[int, int] = field(default_factory=dict)
    calibration_config_path: str | Path | None = DEFAULT_CALIBRATION_CONFIG_PATH
    fov_degrees: float = 90.0
    image_key: str = "image_front"
    segmentation_key: str = "seg_front"
    lidar_key: str = "lidar"


def build_occupancy_preprocessing_config() -> CarlaDataPreprocessingConfig:
    """Return the default preprocessing profile for occupancy work."""

    return CarlaDataPreprocessingConfig()


def build_bc_preprocessing_config(
    *,
    target_image_size: tuple[int, int] = (224, 224),
    normalize_rgb: bool = True,
    hole_fill: bool = True,
) -> CarlaDataPreprocessingConfig:
    """Return the default preprocessing profile for behavioral cloning inputs."""

    return CarlaDataPreprocessingConfig(
        target_image_size=target_image_size,
        normalize_rgb=normalize_rgb,
        hole_fill=hole_fill,
    )


def _load_hf_split(
    dataset_name: str,
    split: str,
    *,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
) -> Any:
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        split=split,
        cache_dir=None if cache_dir is None else str(cache_dir),
        streaming=streaming,
    )


def _image_to_numpy(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must be at least 2D")
    return array


def _rgb_to_tensor(image: Any) -> torch.Tensor:
    rgb = _image_to_numpy(image)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.ndim != 3:
        raise ValueError("RGB image must be HxW or HxWxC")
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[2] != 3:
        raise ValueError("RGB image must have 3 channels")

    rgb = rgb.astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def _segmentation_to_mask(segmentation: Any, *, sample_id: str) -> np.ndarray:
    mask = _image_to_numpy(segmentation)
    if mask.ndim == 2:
        return mask.astype(np.int64)
    if mask.ndim == 3 and mask.shape[2] == 1:
        return mask[..., 0].astype(np.int64)
    if mask.ndim == 3 and mask.shape[2] >= 3:
        rgb = mask[..., :3].astype(np.uint8, copy=False)
        first = rgb[..., 0]
        if np.array_equal(first, rgb[..., 1]) and np.array_equal(first, rgb[..., 2]):
            return first.astype(np.int64)

        decoded = np.full(first.shape, -1, dtype=np.int64)
        for color, class_id in CARLA_CITYSCAPES_PALETTE.items():
            decoded[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
        unknown = decoded < 0
        if np.any(unknown):
            LOGGER.warning(
                "Segmentation sample %s contains %d pixels with unknown colorized labels; "
                "leaving those pixels as -1.",
                sample_id,
                int(np.count_nonzero(unknown)),
            )
        return decoded
    raise ValueError("Segmentation image must be 2D or HxWxC")


def _normalize_lidar(lidar: Any) -> torch.Tensor:
    points = np.asarray(lidar, dtype=np.float32)
    if points.size == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR point cloud must have shape (N, 3+) or (N, 4+)")
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, intensity], axis=1)
    return torch.from_numpy(points[:, :4].copy())


def _resize_tensor(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def _remap_segmentation(mask: np.ndarray, remap: Mapping[int, int]) -> np.ndarray:
    if not remap:
        return mask.astype(np.int64, copy=False)

    remapped = mask.astype(np.int64, copy=True)
    for source, target in remap.items():
        remapped[mask == int(source)] = int(target)
    return remapped


def collate_carla_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate multimodal CARLA samples while keeping variable-length LiDAR as a list."""

    if not batch:
        raise ValueError("batch must contain at least one sample")

    return {
        "rgb": torch.stack([sample["rgb"] for sample in batch], dim=0),
        "depth": torch.stack([sample["depth"] for sample in batch], dim=0),
        "segmentation": torch.stack([sample["segmentation"] for sample in batch], dim=0),
        "control": torch.stack([sample["control"] for sample in batch], dim=0),
        "lidar": [sample["lidar"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }


class CarlaMultimodalDataset(Dataset[dict[str, Any]]):
    """Map-style PyTorch dataset for aligned CARLA multimodal samples."""

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        *,
        preprocessing: CarlaDataPreprocessingConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.preprocessing = preprocessing or CarlaDataPreprocessingConfig()
        self.manual_calibration = load_manual_calibration_config(
            self.preprocessing.calibration_config_path
        )
        self.extrinsics = self._resolve_extrinsics()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image_key = self.preprocessing.image_key
        segmentation_key = self.preprocessing.segmentation_key
        lidar_key = self.preprocessing.lidar_key
        sample_id = f"{sample.get('run_id', 'unknown')}:{sample.get('frame', index)}"

        rgb = _rgb_to_tensor(sample[image_key])
        lidar = _normalize_lidar(sample[lidar_key])
        segmentation_mask = _segmentation_to_mask(sample[segmentation_key], sample_id=sample_id)
        segmentation_mask = _remap_segmentation(
            segmentation_mask,
            self.preprocessing.segmentation_remap,
        )

        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        intrinsics = estimate_front_camera_intrinsics(
            width,
            height,
            self.preprocessing.fov_degrees,
        )
        depth_map = lidar_to_depth(
            lidar.numpy(),
            intrinsics,
            self.extrinsics,
            (height, width),
        )
        if self.preprocessing.hole_fill:
            depth_map = fill_depth_holes(
                depth_map,
                iterations=self.preprocessing.hole_fill_iterations,
                window_size=self.preprocessing.hole_fill_window_size,
            )

        depth = torch.from_numpy(depth_map.astype(np.float32, copy=False)).unsqueeze(0)
        segmentation = torch.from_numpy(segmentation_mask.astype(np.int64, copy=False))

        if self.preprocessing.target_image_size is not None:
            target_size = self.preprocessing.target_image_size
            rgb = _resize_tensor(rgb, target_size, mode="bilinear")
            depth = _resize_tensor(depth, target_size, mode="nearest")
            segmentation = (
                _resize_tensor(
                    segmentation.to(dtype=torch.float32).unsqueeze(0),
                    target_size,
                    mode="nearest",
                )
                .squeeze(0)
                .to(dtype=torch.long)
            )

        if self.preprocessing.normalize_rgb:
            mean = torch.tensor(self.preprocessing.rgb_mean, dtype=rgb.dtype).view(3, 1, 1)
            std = torch.tensor(self.preprocessing.rgb_std, dtype=rgb.dtype).view(3, 1, 1)
            rgb = (rgb - mean) / std

        control = torch.tensor(
            [
                float(sample.get("throttle", 0.0)),
                float(sample.get("steer", 0.0)),
                float(sample.get("brake", 0.0)),
            ],
            dtype=torch.float32,
        )

        metadata = self._build_metadata(sample, index=index, depth_map=depth_map)
        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "control": control,
            "lidar": lidar,
            "metadata": metadata,
        }

    def _resolve_extrinsics(self) -> np.ndarray:
        if self.manual_calibration is None or self.manual_calibration.get("extrinsics") is None:
            raise ValueError(
                "A documented LiDAR-to-camera calibration config is required for "
                "CarlaMultimodalDataset. Set preprocessing.calibration_config_path."
            )
        extrinsics = np.asarray(self.manual_calibration["extrinsics"], dtype=np.float64)
        if extrinsics.shape != (4, 4):
            raise ValueError("Calibration config extrinsics must be a 4x4 matrix")
        return extrinsics

    def _build_metadata(
        self,
        sample: Mapping[str, Any],
        *,
        index: int,
        depth_map: np.ndarray,
    ) -> dict[str, Any]:
        metadata = {
            "index": index,
            "run_id": sample.get("run_id"),
            "frame": sample.get("frame"),
            "timestamp": sample.get("timestamp"),
            "speed_kmh": sample.get("speed_kmh"),
            "map_name": sample.get("map_name"),
            "calibration_source": (
                None
                if self.manual_calibration is None
                else self.manual_calibration.get("extrinsics_source_label")
            ),
            "depth_summary": summarize_depth_map(depth_map),
        }

        for key, value in sample.items():
            if key.startswith("weather_") or key in {
                "vehicles_spawned",
                "walkers_spawned",
                "nearby_vehicles_50m",
                "total_npc_vehicles",
                "total_npc_walkers",
                "duration_seconds",
            }:
                metadata[key] = value

        return metadata


def build_carla_dataset_from_hf(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str,
    preprocessing: CarlaDataPreprocessingConfig | None = None,
    cache_dir: str | Path | None = None,
) -> CarlaMultimodalDataset:
    """Load one HF split and wrap it in the map-style CARLA dataset."""

    dataset = _load_hf_split(dataset_name, split, cache_dir=cache_dir, streaming=False)
    return CarlaMultimodalDataset(dataset, preprocessing=preprocessing)


def validate_processed_dataset(
    dataset: Sequence[Mapping[str, Any]],
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Iterate a processed dataset and summarize depth coverage and malformed samples."""

    total = 0
    success = 0
    malformed = 0
    zero_coverage = 0
    coverage_values: list[float] = []
    error_counts: Counter[str] = Counter()

    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(limit):
        total += 1
        try:
            item = dataset[index]
            depth_summary = item["metadata"]["depth_summary"]
            coverage = float(depth_summary["coverage"])
            coverage_values.append(coverage)
            if coverage <= 0.0:
                zero_coverage += 1
            success += 1
        except Exception as exc:  # pragma: no cover - exercised via tests on concrete errors
            malformed += 1
            error_counts[type(exc).__name__] += 1

    coverage_array = np.asarray(coverage_values, dtype=np.float64)
    return {
        "total_samples": total,
        "success_count": success,
        "malformed_count": malformed,
        "zero_coverage_count": zero_coverage,
        "success_rate": float(success / total) if total else 0.0,
        "mean_depth_coverage": float(np.mean(coverage_array)) if coverage_values else 0.0,
        "min_depth_coverage": float(np.min(coverage_array)) if coverage_values else 0.0,
        "max_depth_coverage": float(np.max(coverage_array)) if coverage_values else 0.0,
        "error_counts": dict(error_counts),
    }
