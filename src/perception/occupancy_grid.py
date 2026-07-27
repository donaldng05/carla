"""Semantic occupancy grid construction from depth and segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from src.transforms import camera_to_vehicle

UNKNOWN_CLASS = -1


@dataclass(frozen=True)
class OccupancyGridSpec:
    """Spatial layout for a vehicle-frame semantic occupancy grid."""

    voxel_size_m: float = 0.25
    x_range_m: tuple[float, float] = (-25.0, 25.0)
    y_range_m: tuple[float, float] = (-25.0, 25.0)
    z_range_m: tuple[float, float] = (-2.5, 2.5)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            _axis_size(self.x_range_m, self.voxel_size_m),
            _axis_size(self.y_range_m, self.voxel_size_m),
            _axis_size(self.z_range_m, self.voxel_size_m),
        )


@dataclass(frozen=True)
class SemanticOccupancyGrid:
    """Discrete occupancy grid with dominant semantic class per occupied voxel."""

    spec: OccupancyGridSpec
    occupied: np.ndarray
    semantic: np.ndarray
    counts: np.ndarray

    def occupied_points(self) -> np.ndarray:
        """Return vehicle-frame centers for occupied voxels."""

        indices = np.argwhere(self.occupied)
        return voxel_indices_to_points(indices, self.spec)


def _axis_size(axis_range: tuple[float, float], voxel_size_m: float) -> int:
    lower, upper = axis_range
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if upper <= lower:
        raise ValueError("axis range upper bound must be greater than lower bound")
    size = (upper - lower) / voxel_size_m
    rounded = int(round(size))
    if not np.isclose(size, rounded):
        raise ValueError("axis range must be evenly divisible by voxel_size_m")
    return rounded


def _validate_intrinsics(K: np.ndarray | list[list[float]]) -> np.ndarray:
    intrinsics = np.asarray(K, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("K must be a 3x3 camera intrinsic matrix")
    if intrinsics[0, 0] == 0.0 or intrinsics[1, 1] == 0.0:
        raise ValueError("K focal lengths must be non-zero")
    return intrinsics


def _validate_depth_and_segmentation(
    depth_map: np.ndarray | list[list[float]],
    segmentation: np.ndarray | list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("depth_map must be a 2D array or 1xHxW array")

    labels = np.asarray(segmentation)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 2:
        raise ValueError("segmentation must be a 2D array or 1xHxW array")
    if labels.shape != depth.shape:
        raise ValueError("depth_map and segmentation must have matching height and width")

    return depth, labels.astype(np.int64, copy=False)


def backproject_depth_to_camera_points(
    depth_map: np.ndarray | list[list[float]],
    K: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject finite positive depth pixels into pinhole camera-frame 3D points."""

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("depth_map must be a 2D array or 1xHxW array")

    intrinsics = _validate_intrinsics(K)
    valid = np.isfinite(depth) & (depth > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    z = depth[rows, cols]
    x = (cols.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.column_stack([x, y, z])
    pixels = np.column_stack([rows, cols])
    return points, pixels


def backproject_depth_to_vehicle_points(
    depth_map: np.ndarray | list[list[float]],
    K: np.ndarray | list[list[float]],
    camera_to_vehicle_extrinsics: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject depth pixels and transform them into the vehicle frame."""

    camera_points, pixels = backproject_depth_to_camera_points(depth_map, K)
    if camera_points.size == 0:
        return camera_points, pixels
    return camera_to_vehicle(camera_points, camera_to_vehicle_extrinsics), pixels


def voxelize_points(
    points: np.ndarray | list[list[float]],
    spec: OccupancyGridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert vehicle-frame points to voxel indices and return the valid point mask."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.zeros((0,), dtype=bool)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    mins = np.array([spec.x_range_m[0], spec.y_range_m[0], spec.z_range_m[0]], dtype=np.float64)
    maxs = np.array([spec.x_range_m[1], spec.y_range_m[1], spec.z_range_m[1]], dtype=np.float64)
    finite = np.all(np.isfinite(point_array), axis=1)
    in_bounds = finite & np.all((point_array >= mins) & (point_array < maxs), axis=1)
    indices = np.floor((point_array[in_bounds] - mins) / spec.voxel_size_m).astype(np.int64)
    return indices, in_bounds


def voxel_indices_to_points(
    indices: np.ndarray | list[list[int]], spec: OccupancyGridSpec
) -> np.ndarray:
    """Return vehicle-frame voxel center points for integer voxel indices."""

    index_array = np.asarray(indices, dtype=np.float64)
    if index_array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if index_array.ndim != 2 or index_array.shape[1] != 3:
        raise ValueError("indices must have shape (N, 3)")
    mins = np.array([spec.x_range_m[0], spec.y_range_m[0], spec.z_range_m[0]], dtype=np.float64)
    return mins + (index_array + 0.5) * spec.voxel_size_m


def build_semantic_occupancy_grid(
    depth_map: np.ndarray | list[list[float]],
    segmentation: np.ndarray | list[list[int]],
    K: np.ndarray | list[list[float]],
    camera_to_vehicle_extrinsics: np.ndarray | list[list[float]],
    *,
    spec: OccupancyGridSpec | None = None,
) -> SemanticOccupancyGrid:
    """Build a semantic occupancy grid from aligned depth and segmentation images."""

    grid_spec = spec or OccupancyGridSpec()
    depth, labels = _validate_depth_and_segmentation(depth_map, segmentation)
    vehicle_points, pixels = backproject_depth_to_vehicle_points(
        depth,
        K,
        camera_to_vehicle_extrinsics,
    )
    shape = grid_spec.shape
    occupied = np.zeros(shape, dtype=bool)
    semantic = np.full(shape, UNKNOWN_CLASS, dtype=np.int16)
    counts = np.zeros(shape, dtype=np.uint16)
    if vehicle_points.size == 0:
        return SemanticOccupancyGrid(grid_spec, occupied, semantic, counts)

    indices, valid_points = voxelize_points(vehicle_points, grid_spec)
    if indices.size == 0:
        return SemanticOccupancyGrid(grid_spec, occupied, semantic, counts)

    valid_pixels = pixels[valid_points]
    point_labels = labels[valid_pixels[:, 0], valid_pixels[:, 1]].astype(np.int64)
    linear = np.ravel_multi_index((indices[:, 0], indices[:, 1], indices[:, 2]), shape)

    flat_counts = np.bincount(linear, minlength=int(np.prod(shape)))
    counts.flat[: flat_counts.size] = np.minimum(flat_counts, np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    occupied.reshape(-1)[:] = flat_counts > 0

    pairs = np.column_stack([linear, point_labels])
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-pair_counts, unique_pairs[:, 0]))
    ordered_pairs = unique_pairs[order]
    _, first_positions = np.unique(ordered_pairs[:, 0], return_index=True)
    winning_pairs = ordered_pairs[first_positions]
    semantic.flat[winning_pairs[:, 0]] = winning_pairs[:, 1].astype(np.int16)

    return SemanticOccupancyGrid(grid_spec, occupied, semantic, counts)


def make_bev_figure(
    grid: SemanticOccupancyGrid,
    *,
    class_colors: Mapping[int, str] | None = None,
    title: str = "Semantic occupancy BEV",
) -> tuple[Any, Any]:
    """Create a top-down semantic occupancy visualization."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise ImportError("matplotlib is required for make_bev_figure") from exc

    colors = {
        UNKNOWN_CLASS: "#ffffff",
        0: "#dddddd",
        1: "#808080",
        2: "#c83f49",
        3: "#f2a23a",
    }
    if class_colors is not None:
        colors.update(class_colors)

    bev = np.full(grid.spec.shape[:2], UNKNOWN_CLASS, dtype=np.int16)
    occupied_xy = np.any(grid.occupied, axis=2)
    for x_idx, y_idx in np.argwhere(occupied_xy):
        z_indices = np.flatnonzero(grid.occupied[x_idx, y_idx])
        if z_indices.size:
            bev[x_idx, y_idx] = grid.semantic[x_idx, y_idx, z_indices[-1]]

    classes = sorted(set(colors) | set(int(v) for v in np.unique(bev)))
    color_values = [colors.get(cls, "#4c78a8") for cls in classes]
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    image = np.vectorize(class_to_index.get)(bev).T

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(
        image,
        origin="lower",
        cmap=mcolors.ListedColormap(color_values),
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Vehicle x voxel")
    ax.set_ylabel("Vehicle y voxel")
    return fig, ax
