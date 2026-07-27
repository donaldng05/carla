import warnings

import numpy as np
import pytest

from src.perception.lidar_conversion import (
    fill_depth_holes,
    lidar_to_depth,
    make_validation_figure,
    project_lidar_to_camera,
)


def test_lidar_to_depth_places_known_point_in_expected_pixel() -> None:
    K = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.eye(4)
    lidar_points = np.array([[0.0, 0.0, 5.0, 0.0]])

    depth = lidar_to_depth(lidar_points, K, extrinsics, (4, 5))

    assert np.isnan(depth).sum() == depth.size - 1
    assert depth[1, 1] == 5.0


def test_lidar_to_depth_keeps_nearest_point_per_pixel() -> None:
    K = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.eye(4)
    lidar_points = np.array(
        [
            [1.0, 0.0, 10.0, 0.0],
            [0.5, 0.0, 5.0, 0.0],
            [0.2, 0.1, 2.0, 0.0],
        ]
    )

    depth = lidar_to_depth(lidar_points, K, extrinsics, (4, 20))

    assert depth[0, 10] == 5.0


def test_lidar_to_depth_ignores_out_of_bounds_and_behind_camera_points() -> None:
    K = np.array(
        [
            [50.0, 0.0, 2.0],
            [0.0, 50.0, 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.eye(4)
    lidar_points = np.array(
        [
            [0.0, 0.0, 2.0, 0.0],  # in bounds
            [100.0, 0.0, 2.0, 0.0],  # out of bounds
            [0.0, 0.0, -1.0, 0.0],  # behind camera
        ]
    )

    depth = lidar_to_depth(lidar_points, K, extrinsics, (5, 5))

    assert np.count_nonzero(~np.isnan(depth)) == 1
    assert depth[2, 2] == 2.0


def test_project_lidar_to_camera_returns_rows_cols_and_depths() -> None:
    K = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.eye(4)
    lidar_points = np.array([[0.0, 0.0, 5.0, 0.0]])

    rows, cols, depths = project_lidar_to_camera(lidar_points, K, extrinsics, (4, 5))

    np.testing.assert_array_equal(rows, np.array([1]))
    np.testing.assert_array_equal(cols, np.array([1]))
    np.testing.assert_allclose(depths, np.array([5.0]))


def test_fill_depth_holes_fills_isolated_nan_with_local_median() -> None:
    depth = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, np.nan, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )

    filled = fill_depth_holes(depth, iterations=1, window_size=3)

    assert np.isfinite(filled[1, 1])
    assert filled[1, 1] == 1.5


def test_fill_depth_holes_keeps_all_nan_windows_quiet() -> None:
    depth = np.full((3, 3), np.nan)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filled = fill_depth_holes(depth, iterations=1, window_size=3)

    assert np.isnan(filled).all()
    assert not any(issubclass(warning.category, RuntimeWarning) for warning in caught)


def test_make_validation_figure_creates_two_axes_without_error() -> None:
    pytest.importorskip("matplotlib")
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    K = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = np.eye(4)
    lidar_points = np.array([[0.0, 0.0, 5.0, 0.0]])

    fig, axes = make_validation_figure(rgb, lidar_points, K, extrinsics)

    assert len(axes) == 2
    fig.clf()
