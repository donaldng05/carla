import numpy as np

from src.transforms import camera_to_vehicle, transform_points, vehicle_to_world, world_to_vehicle


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
