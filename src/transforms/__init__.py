"""Shared coordinate transform helpers for the CARLA project."""

from __future__ import annotations

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


def inverse_transform(transform: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return the inverse of a 4x4 homogeneous transform."""

    matrix = _as_transform_matrix(transform, "transform")
    return np.linalg.inv(matrix)


def compose_transforms(
    first: np.ndarray | list[list[float]],
    second: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Compose two homogeneous transforms so the returned matrix applies first, then second."""

    first_matrix = _as_transform_matrix(first, "first")
    second_matrix = _as_transform_matrix(second, "second")
    return second_matrix @ first_matrix


def carla_pose_to_matrix(
    *,
    x: float,
    y: float,
    z: float,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    degrees: bool = True,
) -> np.ndarray:
    """Build a CARLA/Unreal-style vehicle-to-world homogeneous pose matrix.

    CARLA metadata stores location in meters and rotation as pitch/yaw/roll.
    The matrix applies intrinsic roll, pitch, then yaw rotations in a right-handed
    numeric convention suitable for consistent relative-frame testing.
    """

    angles = np.array([roll, pitch, yaw], dtype=np.float64)
    if degrees:
        angles = np.deg2rad(angles)
    roll_rad, pitch_rad, yaw_rad = angles

    cr = float(np.cos(roll_rad))
    sr = float(np.sin(roll_rad))
    cp = float(np.cos(pitch_rad))
    sp = float(np.sin(pitch_rad))
    cy = float(np.cos(yaw_rad))
    sy = float(np.sin(yaw_rad))

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float64)
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

    return transform_points(points, inverse_transform(ego_pose))
