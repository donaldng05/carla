import numpy as np
import pytest

from src.perception.occupancy_grid import (
    UNKNOWN_CLASS,
    OccupancyGridSpec,
    backproject_depth_to_camera_points,
    build_semantic_occupancy_grid,
    make_bev_figure,
    voxel_indices_to_points,
    voxelize_points,
)


def small_spec() -> OccupancyGridSpec:
    return OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(-2.0, 4.0),
        y_range_m=(-3.0, 3.0),
        z_range_m=(0.0, 6.0),
    )


def test_backprojection_uses_pinhole_formula_and_ignores_nan() -> None:
    depth = np.array([[np.nan, 2.0], [4.0, -1.0]])
    K = np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]])

    points, pixels = backproject_depth_to_camera_points(depth, K)

    np.testing.assert_allclose(points, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 4.0]]))
    np.testing.assert_array_equal(pixels, np.array([[0, 1], [1, 0]]))


def test_voxelize_points_returns_indices_and_valid_mask() -> None:
    points = np.array([[0.25, -0.25, 1.2], [9.0, 0.0, 0.0]])

    indices, valid = voxelize_points(points, small_spec())

    np.testing.assert_array_equal(indices, np.array([[2, 2, 1]]))
    np.testing.assert_array_equal(valid, np.array([True, False]))


def test_voxel_indices_to_points_returns_voxel_centers() -> None:
    points = voxel_indices_to_points(np.array([[2, 2, 1]]), small_spec())

    np.testing.assert_allclose(points, np.array([[0.5, -0.5, 1.5]]))


def test_build_semantic_occupancy_grid_places_known_pixel_in_expected_voxel() -> None:
    depth = np.full((3, 3), np.nan)
    depth[1, 1] = 2.0
    segmentation = np.zeros((3, 3), dtype=np.int64)
    segmentation[1, 1] = 7
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    grid = build_semantic_occupancy_grid(depth, segmentation, K, np.eye(4), spec=small_spec())

    assert grid.occupied[2, 3, 2]
    assert grid.semantic[2, 3, 2] == 7
    assert grid.counts[2, 3, 2] == 1
    assert np.count_nonzero(grid.occupied) == 1


def test_build_semantic_occupancy_grid_uses_dominant_class_per_voxel() -> None:
    depth = np.array([[2.0, 2.0, 2.0]])
    segmentation = np.array([[4, 4, 9]])
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    extrinsics = np.eye(4)
    extrinsics[0, 3] = 0.5

    grid = build_semantic_occupancy_grid(depth, segmentation, K, extrinsics, spec=small_spec())

    assert np.count_nonzero(grid.occupied) == 1
    occupied_index = tuple(np.argwhere(grid.occupied)[0])
    assert grid.semantic[occupied_index] == 4
    assert grid.counts[occupied_index] == 3


def test_empty_depth_returns_unknown_empty_grid() -> None:
    depth = np.full((3, 3), np.nan)
    segmentation = np.zeros((3, 3), dtype=np.int64)
    K = np.eye(3)

    grid = build_semantic_occupancy_grid(depth, segmentation, K, np.eye(4), spec=small_spec())

    assert not grid.occupied.any()
    assert np.all(grid.semantic == UNKNOWN_CLASS)


def test_make_bev_figure_creates_axis_without_error() -> None:
    pytest.importorskip("matplotlib")
    depth = np.full((3, 3), np.nan)
    depth[1, 1] = 2.0
    segmentation = np.ones((3, 3), dtype=np.int64)
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    grid = build_semantic_occupancy_grid(depth, segmentation, K, np.eye(4), spec=small_spec())

    fig, ax = make_bev_figure(grid)

    assert ax.get_title() == "Semantic occupancy BEV"
    fig.clf()
