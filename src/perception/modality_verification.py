"""Helpers for dataset-level modality verification and notebook exploration."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

INTRINSIC_KEYS: tuple[str, ...] = (
    "camera_intrinsics",
    "K",
    "front_camera_intrinsics",
)
EXTRINSIC_KEYS: tuple[str, ...] = (
    "lidar_to_camera_extrinsics",
    "extrinsics",
    "camera_extrinsics",
)
DATASET_DERIVED_INTRINSICS_LABEL = "dataset-derived"


def _image_to_shape(image: Any) -> tuple[int, int]:
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        width, height = image.size
        return int(height), int(width)

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must be at least 2D")
    height, width = array.shape[:2]
    return int(height), int(width)


def estimate_front_camera_intrinsics(
    width: int, height: int, fov_degrees: float = 90.0
) -> np.ndarray:
    """Estimate a pinhole camera intrinsic matrix from image size and horizontal FOV."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not (0.0 < fov_degrees < 180.0):
        raise ValueError("fov_degrees must be between 0 and 180")

    focal = width / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resolve_front_camera_intrinsics(
    sample: Mapping[str, Any],
    *,
    image_key: str = "image_front",
    fov_degrees: float = 90.0,
) -> np.ndarray:
    """Load front-camera intrinsics from a sample or derive them from the image size."""

    for key in INTRINSIC_KEYS:
        if key in sample and sample[key] is not None:
            matrix = np.asarray(sample[key], dtype=np.float64)
            if matrix.shape == (3, 3):
                return matrix
            raise ValueError(f"{key} must be a 3x3 matrix")

    image_shape = _image_to_shape(sample[image_key])
    return estimate_front_camera_intrinsics(image_shape[1], image_shape[0], fov_degrees)


def resolve_lidar_to_camera_extrinsics(
    sample: Mapping[str, Any],
    *,
    keys: Sequence[str] = EXTRINSIC_KEYS,
) -> np.ndarray:
    """Load the LiDAR-to-camera transform from a sample."""

    for key in keys:
        if key in sample and sample[key] is not None:
            matrix = np.asarray(sample[key], dtype=np.float64)
            if matrix.shape == (4, 4):
                return matrix
            raise ValueError(f"{key} must be a 4x4 matrix")

    raise KeyError(
        "No LiDAR-to-camera extrinsics found in the sample. "
        "Check dataset metadata or add the calibration to the notebook config."
    )


def load_manual_calibration_config(
    path: str | Path | None,
) -> dict[str, Any] | None:
    """Load an optional JSON calibration config for notebook inspection."""

    if path is None:
        return None

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manual calibration config must be a JSON object")

    calibration: dict[str, Any] = {
        "path": str(config_path),
        "provenance": payload.get("provenance", f"manual config at {config_path}"),
    }
    if "intrinsics_source_label" in payload:
        calibration["intrinsics_source_label"] = str(payload["intrinsics_source_label"])
    if "extrinsics_source_label" in payload:
        calibration["extrinsics_source_label"] = str(payload["extrinsics_source_label"])

    for key in INTRINSIC_KEYS:
        if key in payload and payload[key] is not None:
            matrix = np.asarray(payload[key], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"{key} in manual calibration config must be a 3x3 matrix")
            calibration["intrinsics"] = matrix
            break

    for key in EXTRINSIC_KEYS:
        if key in payload and payload[key] is not None:
            matrix = np.asarray(payload[key], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{key} in manual calibration config must be a 4x4 matrix")
            calibration["extrinsics"] = matrix
            break

    return calibration


def discover_calibration_fields(
    sample: Mapping[str, Any],
    *,
    key_hints: Sequence[str] = (
        "intrinsic",
        "extrinsic",
        "camera",
        "lidar",
        "calib",
        "rotation",
        "translation",
        "transform",
        "pose",
        "fov",
        "sensor",
    ),
) -> dict[str, str]:
    """Return sample fields that look calibration-related.

    The output is a lightweight mapping of field name to a short type/shape summary
    so notebook inspection can show what calibration-like metadata exists.
    """

    discovered: dict[str, str] = {}
    hints = tuple(hint.lower() for hint in key_hints)
    stack: list[tuple[str, Any]] = [(str(key), value) for key, value in sample.items()]

    while stack:
        key, value = stack.pop()
        key_lower = key.lower()
        if any(hint in key_lower for hint in hints):
            try:
                array = np.asarray(value)
            except Exception:
                array = None

            if array is not None and array.ndim > 0:
                discovered[key] = f"array{tuple(int(dim) for dim in array.shape)}"
            elif isinstance(value, Mapping):
                discovered[key] = f"mapping[{len(value)}]"
            else:
                discovered[key] = type(value).__name__

        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                stack.append((f"{key}.{child_key}", child_value))

    return discovered


def resolve_calibration_status(
    sample: Mapping[str, Any],
    *,
    image_key: str = "image_front",
    fov_degrees: float = 90.0,
    manual_intrinsics: np.ndarray | None = None,
    manual_extrinsics: np.ndarray | None = None,
    manual_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve calibration availability and report whether projection is trustworthy."""

    calibration_fields = discover_calibration_fields(sample)

    intrinsics: np.ndarray | None = None
    intrinsics_source = "missing"
    intrinsics_available = False
    blocking_reason: str | None = None

    for key in INTRINSIC_KEYS:
        if key in sample and sample[key] is not None:
            intrinsics = np.asarray(sample[key], dtype=np.float64)
            if intrinsics.shape != (3, 3):
                raise ValueError(f"{key} must be a 3x3 matrix")
            intrinsics_source = "sample metadata"
            intrinsics_available = True
            break

    if intrinsics is None and manual_intrinsics is not None:
        intrinsics = np.asarray(manual_intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError("manual_intrinsics must be a 3x3 matrix")
        intrinsics_source = "manual override"
        intrinsics_available = True

    if (
        intrinsics is None
        and manual_calibration is not None
        and manual_calibration.get("intrinsics") is not None
    ):
        intrinsics = np.asarray(manual_calibration["intrinsics"], dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError("manual calibration intrinsics must be a 3x3 matrix")
        provenance = manual_calibration.get("provenance", "manual calibration config")
        intrinsics_source = str(
            manual_calibration.get(
                "intrinsics_source_label",
                f"manual calibration config ({provenance})",
            )
        )
        intrinsics_available = True

    if intrinsics is None:
        intrinsics = resolve_front_camera_intrinsics(
            sample,
            image_key=image_key,
            fov_degrees=fov_degrees,
        )
        intrinsics_source = f"{DATASET_DERIVED_INTRINSICS_LABEL} (image shape + FOV={fov_degrees})"
        intrinsics_available = True

    extrinsics: np.ndarray | None = None
    extrinsics_source = "missing"
    extrinsics_available = False

    for key in EXTRINSIC_KEYS:
        if key in sample and sample[key] is not None:
            extrinsics = np.asarray(sample[key], dtype=np.float64)
            if extrinsics.shape != (4, 4):
                raise ValueError(f"{key} must be a 4x4 matrix")
            extrinsics_source = "sample metadata"
            extrinsics_available = True
            break

    if extrinsics is None and manual_extrinsics is not None:
        extrinsics = np.asarray(manual_extrinsics, dtype=np.float64)
        if extrinsics.shape != (4, 4):
            raise ValueError("manual_extrinsics must be a 4x4 matrix")
        extrinsics_source = "manual override"
        extrinsics_available = True

    if (
        extrinsics is None
        and manual_calibration is not None
        and manual_calibration.get("extrinsics") is not None
    ):
        extrinsics = np.asarray(manual_calibration["extrinsics"], dtype=np.float64)
        if extrinsics.shape != (4, 4):
            raise ValueError("manual calibration extrinsics must be a 4x4 matrix")
        provenance = manual_calibration.get("provenance", "manual calibration config")
        extrinsics_source = str(
            manual_calibration.get(
                "extrinsics_source_label",
                f"manual calibration config ({provenance})",
            )
        )
        extrinsics_available = True

    if not extrinsics_available:
        blocking_reason = "LiDAR-to-camera extrinsics are unavailable; projection and dense depth validation remain blocked."

    can_project = intrinsics_available and extrinsics_available
    status = (
        f"projection enabled via {intrinsics_source} + {extrinsics_source}"
        if can_project
        else f"projection blocked: {blocking_reason}"
    )

    return {
        "can_project": can_project,
        "intrinsics_available": intrinsics_available,
        "extrinsics_available": extrinsics_available,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "intrinsics_source": intrinsics_source,
        "extrinsics_source": extrinsics_source,
        "blocking_reason": blocking_reason,
        "status": status,
        "calibration_fields": calibration_fields,
        "fields_checked": {
            "intrinsics": list(INTRINSIC_KEYS),
            "extrinsics": list(EXTRINSIC_KEYS),
        },
    }


def resolve_projection_inputs(
    sample: Mapping[str, Any],
    *,
    image_key: str = "image_front",
    fov_degrees: float = 90.0,
    manual_intrinsics: np.ndarray | None = None,
    manual_extrinsics: np.ndarray | None = None,
    manual_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve projection inputs and report whether depth projection is possible."""

    return resolve_calibration_status(
        sample,
        image_key=image_key,
        fov_degrees=fov_degrees,
        manual_intrinsics=manual_intrinsics,
        manual_extrinsics=manual_extrinsics,
        manual_calibration=manual_calibration,
    )


def summarize_depth_map(depth_map: np.ndarray | list[list[float]]) -> dict[str, float]:
    """Return simple coverage statistics for a dense depth map."""

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be 2D")

    finite = np.isfinite(depth)
    finite_count = int(finite.sum())
    total = int(depth.size)
    coverage = float(finite_count / total) if total else 0.0
    if finite_count:
        values = depth[finite]
        return {
            "finite_count": float(finite_count),
            "total_count": float(total),
            "coverage": coverage,
            "min_depth": float(np.min(values)),
            "max_depth": float(np.max(values)),
            "mean_depth": float(np.mean(values)),
        }

    return {
        "finite_count": 0.0,
        "total_count": float(total),
        "coverage": 0.0,
        "min_depth": float("nan"),
        "max_depth": float("nan"),
        "mean_depth": float("nan"),
    }


def summarize_sample(
    sample: Mapping[str, Any],
    *,
    image_key: str = "image_front",
    lidar_key: str = "lidar",
    seg_key: str = "seg_front",
) -> dict[str, Any]:
    """Summarize one dataset sample for notebook inspection."""

    image = sample.get(image_key)
    lidar = sample.get(lidar_key)
    segmentation = sample.get(seg_key)

    summary: dict[str, Any] = {
        "run_id": sample.get("run_id"),
        "frame": sample.get("frame"),
        "timestamp": sample.get("timestamp"),
        "speed_kmh": sample.get("speed_kmh"),
        "has_image": image is not None,
        "has_lidar": lidar is not None,
        "has_segmentation": segmentation is not None,
        "has_boxes": sample.get("boxes") is not None,
        "has_box_labels": sample.get("box_labels") is not None,
        "available_keys": sorted(sample.keys()),
    }

    if image is not None:
        summary["image_height"], summary["image_width"] = _image_to_shape(image)

    if lidar is not None:
        lidar_array = np.asarray(lidar)
        summary["lidar_points"] = (
            int(lidar_array.shape[0]) if lidar_array.ndim >= 2 else int(lidar_array.size // 4)
        )

    if segmentation is not None:
        summary["segmentation_height"], summary["segmentation_width"] = _image_to_shape(
            segmentation
        )

    summary["calibration_keys_present"] = sorted(discover_calibration_fields(sample).keys())
    return summary


def take_stream_window(
    dataset: Iterable[Mapping[str, Any]],
    *,
    target_count: int = 12,
) -> list[Mapping[str, Any]]:
    """Take the first N samples from an iterable dataset and stop immediately."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")

    selected: list[Mapping[str, Any]] = []
    for sample in dataset:
        selected.append(sample)
        if len(selected) >= target_count:
            break

    return selected


def select_representative_samples(
    dataset: Iterable[Mapping[str, Any]],
    *,
    target_count: int = 12,
    per_run: int = 2,
    run_key: str = "run_id",
) -> list[Mapping[str, Any]]:
    """Select a small sample set spread across runs."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if per_run <= 0:
        raise ValueError("per_run must be positive")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in dataset:
        run_id = str(sample.get(run_key, "unknown"))
        if len(grouped[run_id]) < per_run:
            grouped[run_id].append(sample)

        if sum(len(items) for items in grouped.values()) >= target_count:
            break

    selected: list[Mapping[str, Any]] = []
    for run_id in sorted(grouped):
        selected.extend(grouped[run_id])
        if len(selected) >= target_count:
            return selected[:target_count]

    return selected[:target_count]
