import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.transforms import (
    camera_to_vehicle,
    compose_transforms,
    inverse_transform,
    transform_points,
    vehicle_to_world,
    world_to_vehicle,
)


def test_transform_points_identity_leaves_points_unchanged() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.5, 0.25, 9.75]])
    transform = np.eye(4)

    transformed = transform_points(points, transform)

    np.testing.assert_allclose(transformed, points)


def test_transform_points_applies_translation() -> None:
    points = np.array([1.0, 2.0, 3.0])
    transform = np.eye(4)
    transform[:3, 3] = np.array([10.0, -2.0, 0.5])

    transformed = transform_points(points, transform)

    np.testing.assert_allclose(transformed, np.array([11.0, 0.0, 3.5]))


def test_vehicle_world_round_trip_recovers_original_points() -> None:
    points = np.array([[0.2, -0.1, 3.5], [2.0, 1.5, -4.0]])
    ego_pose = np.eye(4)
    ego_pose[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ego_pose[:3, 3] = np.array([4.0, 5.0, 6.0])

    world = vehicle_to_world(points, ego_pose)
    recovered = world_to_vehicle(world, ego_pose)

    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_camera_to_vehicle_wrapper_uses_same_homogeneous_transform() -> None:
    point = np.array([1.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed = camera_to_vehicle(point, transform)

    np.testing.assert_allclose(transformed, np.array([0.0, 1.0, 0.0]))


def test_known_90_degree_z_rotation_maps_x_axis_to_y_axis() -> None:
    point = np.array([1.0, 0.0, 0.0])
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transformed = transform_points(point, transform)

    np.testing.assert_allclose(transformed, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_composed_transforms_match_sequential_application() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    first = np.eye(4)
    first[:3, 3] = np.array([3.0, 0.0, 1.0])
    second = np.eye(4)
    second[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sequential = transform_points(transform_points(points, first), second)
    composed = transform_points(points, compose_transforms(first, second))

    np.testing.assert_allclose(composed, sequential, atol=1e-12)


@st.composite
def rigid_transform_strategy(draw: st.DrawFn) -> np.ndarray:
    yaw = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False))
    tx = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
    ty = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False))
    tz = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = np.array([tx, ty, tz])
    return transform


point_strategy = st.tuples(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
).map(lambda values: np.array(values, dtype=np.float64))


@given(point=point_strategy, transform=rigid_transform_strategy())
def test_inverse_transform_property_recovers_original_point(
    point: np.ndarray,
    transform: np.ndarray,
) -> None:
    transformed = transform_points(point, transform)
    recovered = transform_points(transformed, inverse_transform(transform))

    np.testing.assert_allclose(recovered, point, atol=1e-8)


@given(point=point_strategy, transform=rigid_transform_strategy())
def test_rotation_component_preserves_distance(point: np.ndarray, transform: np.ndarray) -> None:
    rotation_only = transform.copy()
    rotation_only[:3, 3] = 0.0

    transformed = transform_points(point, rotation_only)

    np.testing.assert_allclose(np.linalg.norm(transformed), np.linalg.norm(point), atol=1e-8)
