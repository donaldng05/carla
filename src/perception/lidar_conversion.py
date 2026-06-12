"""LiDAR to dense depth conversion utilities."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.transforms import transform_points


def _validate_intrinsics(K: np.ndarray | list[list[float]]) -> np.ndarray:
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("K must be a 3x3 camera intrinsic matrix")
    return intrinsics


def _validate_image_shape(image_shape: Iterable[int]) -> tuple[int, int]:
    shape = tuple(int(v) for v in image_shape)
    if len(shape) != 2:
        raise ValueError("image_shape must be a (height, width) pair")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


def _normalize_lidar_points(
    lidar_points: np.ndarray | list[list[float]] | list[float],
) -> np.ndarray:
    points = np.asarray(lidar_points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    if points.ndim == 1:
        if points.shape[0] < 3:
            raise ValueError("lidar_points must have at least 3 coordinates per point")
        return points[:3][None, :]

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("lidar_points must have shape (N, 3+) or (3+,)")

    return points[:, :3]


def project_lidar_to_camera(
    lidar_points: np.ndarray | list[list[float]] | list[float],
    K: np.ndarray | list[list[float]],
    extrinsics: np.ndarray | list[list[float]],
    image_shape: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points into image coordinates.

    Returns:
        rows: integer row indices in the image
        cols: integer column indices in the image
        depth: camera-frame depth values for the valid projected points
    """

    intrinsics = _validate_intrinsics(K)
    height, width = _validate_image_shape(image_shape)
    points = _normalize_lidar_points(lidar_points)
    if points.size == 0:
        empty_int = np.empty((0,), dtype=np.int64)
        empty_float = np.empty((0,), dtype=np.float64)
        return empty_int, empty_int, empty_float

    camera_points = transform_points(points, extrinsics)
    if camera_points.ndim == 1:
        camera_points = camera_points[None, :]

    depth = camera_points[:, 2]
    finite = np.isfinite(depth) & (depth > 1e-8)
    if not np.any(finite):
        empty_int = np.empty((0,), dtype=np.int64)
        empty_float = np.empty((0,), dtype=np.float64)
        return empty_int, empty_int, empty_float

    camera_points = camera_points[finite]
    depth = depth[finite]

    x = camera_points[:, 0]
    y = camera_points[:, 1]

    u = intrinsics[0, 0] * x / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * y / depth + intrinsics[1, 2]

    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        empty_int = np.empty((0,), dtype=np.int64)
        empty_float = np.empty((0,), dtype=np.float64)
        return empty_int, empty_int, empty_float

    u = u[valid]
    v = v[valid]
    depth = depth[valid]

    cols = np.floor(u + 0.5).astype(np.int64)
    rows = np.floor(v + 0.5).astype(np.int64)

    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.any(in_bounds):
        empty_int = np.empty((0,), dtype=np.int64)
        empty_float = np.empty((0,), dtype=np.float64)
        return empty_int, empty_int, empty_float

    rows = rows[in_bounds]
    cols = cols[in_bounds]
    depth = depth[in_bounds]
    return rows, cols, depth


def lidar_to_depth(
    lidar_points: np.ndarray | list[list[float]] | list[float],
    K: np.ndarray | list[list[float]],
    extrinsics: np.ndarray | list[list[float]],
    image_shape: Iterable[int],
) -> np.ndarray:
    """Convert LiDAR points into a dense front-camera depth map.

    The returned depth map uses camera-frame Z values and preserves
    unobserved pixels as NaN so downstream code can distinguish missing
    observations from far-range returns.
    """

    height, width = _validate_image_shape(image_shape)
    depth_map = np.full((height, width), np.nan, dtype=np.float64)

    rows, cols, depths = project_lidar_to_camera(lidar_points, K, extrinsics, (height, width))
    if depths.size == 0:
        return depth_map

    linear_indices = rows * width + cols
    order = np.lexsort((np.arange(depths.size), depths))
    linear_indices = linear_indices[order]
    depths = depths[order]

    _, first_positions = np.unique(linear_indices, return_index=True)
    selected = first_positions
    depth_map.flat[linear_indices[selected]] = depths[selected]
    return depth_map


def fill_depth_holes(
    depth_map: np.ndarray | list[list[float]],
    *,
    iterations: int = 1,
    window_size: int = 3,
) -> np.ndarray:
    """Optionally fill NaN holes in a depth map using local nan-median filtering."""

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be a 2D array")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    from numpy.lib.stride_tricks import sliding_window_view

    filled = depth.copy()
    radius = window_size // 2
    for _ in range(iterations):
        if not np.isnan(filled).any():
            break

        padded = np.pad(filled, radius, mode="constant", constant_values=np.nan)
        windows = sliding_window_view(padded, (window_size, window_size))
        local_values = np.nanmedian(windows, axis=(-1, -2))
        nan_mask = np.isnan(filled)
        if not np.any(nan_mask):
            break
        filled[nan_mask] = local_values[nan_mask]

    return filled


def make_validation_figure(
    rgb_image: np.ndarray | list,
    lidar_points: np.ndarray | list[list[float]] | list[float],
    K: np.ndarray | list[list[float]],
    extrinsics: np.ndarray | list[list[float]],
    image_shape: Iterable[int] | None = None,
    depth_map: np.ndarray | list[list[float]] | None = None,
    *,
    point_size: float = 3.0,
) -> tuple[Any, Any]:
    """Create a quick visual sanity-check figure for LiDAR projection."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise ImportError("matplotlib is required for make_validation_figure") from exc

    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("rgb_image must be an HxWx3 or HxWx4 array")

    if image_shape is None:
        image_shape = rgb.shape[:2]

    rows, cols, depths = project_lidar_to_camera(lidar_points, K, extrinsics, image_shape)
    if depth_map is None:
        depth_map = lidar_to_depth(lidar_points, K, extrinsics, image_shape)
    else:
        depth_map = np.asarray(depth_map, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].imshow(rgb.astype(np.uint8) if np.issubdtype(rgb.dtype, np.integer) else rgb)
    if depths.size:
        scatter = axes[0].scatter(
            cols, rows, c=depths, s=point_size, cmap="turbo", alpha=0.9, linewidths=0
        )
        fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="Depth (m)")
    axes[0].set_title("RGB with projected LiDAR")
    axes[0].axis("off")

    image = axes[1].imshow(depth_map, cmap="magma")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")
    axes[1].set_title("Dense depth map")
    axes[1].axis("off")

    return fig, axes
