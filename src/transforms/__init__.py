"""Shared coordinate transform helpers for the CARLA project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_points_array(
    points: np.ndarray | list[list[float]] | list[float],
) -> tuple[np.ndarray, bool]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] < 3:
            raise ValueError("points must have at least 3 coordinates")
        return array[:3][None, :], True

    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+) or (3+ ,)")

    return array[:, :3], False


def _as_transform_matrix(transform: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform")
    return matrix


def transform_points(
    points: np.ndarray | list[list[float]] | list[float], transform: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to one or more 3D points."""

    point_array, squeeze = _as_points_array(points)
    matrix = _as_transform_matrix(transform, "transform")

    homogeneous = np.concatenate(
        [point_array, np.ones((point_array.shape[0], 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ matrix.T
    result = transformed[:, :3]
    if squeeze:
        return result[0]
    return result


def camera_to_vehicle(
    points: np.ndarray | list[list[float]] | list[float], extrinsics: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Transform camera-frame points into vehicle frame using a camera-to-vehicle matrix."""

    return transform_points(points, extrinsics)


def vehicle_to_world(
    points: np.ndarray | list[list[float]] | list[float], ego_pose: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Transform vehicle-frame points into world frame using a vehicle-to-world pose matrix."""

    return transform_points(points, ego_pose)


def world_to_vehicle(
    points: np.ndarray | list[list[float]] | list[float], ego_pose: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Transform world-frame points into vehicle frame using a vehicle-to-world pose matrix."""

    matrix = _as_transform_matrix(ego_pose, "ego_pose")
    return transform_points(points, np.linalg.inv(matrix))
