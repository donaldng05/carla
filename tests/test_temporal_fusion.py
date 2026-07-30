import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.perception.occupancy_grid import OccupancyGridSpec, SemanticOccupancyGrid
from src.perception.temporal_fusion import (
    OccupancyFrame,
    TemporalFusionBuffer,
    fuse_occupancy_frames,
    make_temporal_alignment_figure,
    transform_grid_to_ego_frame,
)


def spec() -> OccupancyGridSpec:
    return OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(-3.0, 3.0),
        y_range_m=(-3.0, 3.0),
        z_range_m=(0.0, 3.0),
    )


def grid_with_voxels(voxels: list[tuple[int, int, int, int]]) -> SemanticOccupancyGrid:
    grid_spec = spec()
    occupied = np.zeros(grid_spec.shape, dtype=bool)
    semantic = np.full(grid_spec.shape, -1, dtype=np.int16)
    counts = np.zeros(grid_spec.shape, dtype=np.uint16)
    for x, y, z, label in voxels:
        occupied[x, y, z] = True
        semantic[x, y, z] = label
        counts[x, y, z] = 1
    return SemanticOccupancyGrid(grid_spec, occupied, semantic, counts)


def pose(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.array([x, y, 0.0])
    return matrix


def test_temporal_buffer_keeps_only_five_frames() -> None:
    buffer = TemporalFusionBuffer()

    for i in range(7):
        buffer.add(OccupancyFrame(grid_with_voxels([(3, 3, 1, i)]), pose(float(i))))

    assert len(buffer) == 5
    assert buffer.frames[0].metadata is None


def test_identity_pose_fusion_preserves_current_voxel_score() -> None:
    frame = OccupancyFrame(grid_with_voxels([(3, 3, 1, 2)]), pose())

    fused = fuse_occupancy_frames([frame], weights=(1.0, 0.8))

    assert fused.occupancy_score[3, 3, 1] == 1.0
    assert fused.semantic[3, 3, 1] == 2


def test_previous_grid_is_transformed_into_current_ego_frame() -> None:
    previous = OccupancyFrame(grid_with_voxels([(3, 3, 1, 4)]), pose(0.0))
    current = OccupancyFrame(grid_with_voxels([]), pose(1.0))

    indices, labels, _ = transform_grid_to_ego_frame(
        previous.grid,
        source_ego_pose=previous.ego_pose,
        target_ego_pose=current.ego_pose,
    )

    np.testing.assert_array_equal(indices, np.array([[2, 3, 1]]))
    np.testing.assert_array_equal(labels, np.array([4]))


def test_fusion_current_frame_has_stronger_weight_than_previous() -> None:
    current = OccupancyFrame(grid_with_voxels([(3, 3, 1, 1)]), pose())
    previous = OccupancyFrame(grid_with_voxels([(3, 3, 1, 2)]), pose())

    fused = fuse_occupancy_frames([current, previous], weights=(1.0, 0.8))

    assert fused.semantic[3, 3, 1] == 1
    assert np.isclose(fused.occupancy_score[3, 3, 1], 1.0)


@given(
    current=st.booleans(),
    previous=st.booleans(),
    w0=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
    w1=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
)
def test_fused_occupancy_score_is_bounded(
    current: bool,
    previous: bool,
    w0: float,
    w1: float,
) -> None:
    current_voxels = [(3, 3, 1, 1)] if current else []
    previous_voxels = [(3, 3, 1, 1)] if previous else []
    fused = fuse_occupancy_frames(
        [
            OccupancyFrame(grid_with_voxels(current_voxels), pose()),
            OccupancyFrame(grid_with_voxels(previous_voxels), pose()),
        ],
        weights=(w0, w1),
    )

    assert 0.0 <= fused.occupancy_score[3, 3, 1] <= 1.0


def test_make_temporal_alignment_figure_creates_axis_without_error() -> None:
    pytest.importorskip("matplotlib")
    current = OccupancyFrame(grid_with_voxels([(3, 3, 1, 1)]), pose())
    previous = OccupancyFrame(grid_with_voxels([(3, 3, 1, 1)]), pose())

    fig, ax = make_temporal_alignment_figure(current, previous)

    assert ax.get_title() == "Temporal occupancy alignment"
    fig.clf()
