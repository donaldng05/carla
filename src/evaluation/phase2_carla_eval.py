"""Bounded streamed CARLA Phase 2 occupancy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.data.carla_dataset import (
    DEFAULT_DATASET_NAME,
    CarlaDataPreprocessingConfig,
    CarlaMultimodalDataset,
    build_occupancy_preprocessing_config,
)
from src.evaluation.iou import (
    ClassIoU,
    compute_semantic_iou,
    iou_table,
    summarize_iou,
    write_iou_summary_json,
)
from src.evaluation.shadow_mode import ShadowModeEvaluator, occupancy_disagreement_rate
from src.perception.modality_verification import estimate_front_camera_intrinsics
from src.perception.occupancy_grid import (
    UNKNOWN_CLASS,
    OccupancyGridSpec,
    SemanticOccupancyGrid,
    build_semantic_occupancy_grid,
)
from src.perception.temporal_fusion import (
    DEFAULT_FUSION_WEIGHTS,
    FusedOccupancyGrid,
    OccupancyFrame,
    TemporalFusionBuffer,
    fuse_occupancy_frames,
    transform_grid_to_ego_frame,
)
from src.transforms import carla_pose_to_matrix

ROAD_CLASS_ID = 7
VEHICLE_CLASS_ID = 10
PEDESTRIAN_CLASS_ID = 4

CARLA_CLASS_NAMES = {
    ROAD_CLASS_ID: "road",
    VEHICLE_CLASS_ID: "vehicle",
    PEDESTRIAN_CLASS_ID: "pedestrian",
}
DEFAULT_CLASS_IDS = tuple(CARLA_CLASS_NAMES)
BEV_CLASS_NAMES = {
    1: "building",
    4: "pedestrian",
    6: "road line",
    7: "road",
    8: "sidewalk",
    10: "vehicle",
}
BEV_CLASS_COLORS = {
    UNKNOWN_CLASS: "#ffffff",
    1: "#b9c7d8",
    4: "#f2a23a",
    6: "#6aa5ff",
    7: "#777777",
    8: "#c7c7c7",
    10: "#c83f49",
}


@dataclass(frozen=True)
class Phase2CarlaEvalConfig:
    """Configuration for a bounded streamed CARLA occupancy evaluation."""

    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "validation"
    max_frames: int = 200
    output_dir: str | Path = "outputs/phase2"
    occupancy_threshold: float = 0.25
    disagreement_threshold: float = 0.05
    bev_frame_count: int = 5
    preprocessing: CarlaDataPreprocessingConfig | None = None
    grid_spec: OccupancyGridSpec | None = None
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS
    class_names: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        if self.max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if self.bev_frame_count < 0:
            raise ValueError("bev_frame_count must be non-negative")
        if self.class_names is None:
            object.__setattr__(self, "class_names", CARLA_CLASS_NAMES)


@dataclass(frozen=True)
class Phase2CarlaEvalArtifacts:
    """Paths written by the bounded Phase 2 CARLA evaluation."""

    baseline_iou: Path
    temporal_iou: Path
    shadow_records: Path
    shadow_clusters: Path
    run_summary: Path
    bev_images: tuple[Path, ...]


@dataclass(frozen=True)
class FusionProfile:
    """Named temporal-fusion weighting profile for ablation runs."""

    name: str
    weights: tuple[float, ...]


DEFAULT_STRICT_FUSION_WEIGHTS = (1.0, 0.5, 0.25, 0.1, 0.05)


@dataclass(frozen=True)
class FuturePseudoEvalConfig:
    """Configuration for fairer future pseudo-label Phase 2 evaluation."""

    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "validation"
    anchor_count: int = 200
    output_dir: str | Path = "outputs/phase2"
    occupancy_threshold: float = 0.25
    disagreement_threshold: float = 0.05
    bev_frame_count: int = 5
    alignment_frame_count: int = 3
    alignment_tolerance_voxels: int = 2
    past_window: int = 5
    future_window: int = 5
    thresholds: tuple[float, ...] = (0.25, 0.4, 0.5, 0.6)
    fusion_profiles: tuple[FusionProfile, ...] = (
        FusionProfile("current_weights", DEFAULT_FUSION_WEIGHTS),
        FusionProfile("strict_decay", DEFAULT_STRICT_FUSION_WEIGHTS),
    )
    preprocessing: CarlaDataPreprocessingConfig | None = None
    grid_spec: OccupancyGridSpec | None = None
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS
    class_names: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        if self.anchor_count < 1:
            raise ValueError("anchor_count must be at least 1")
        if self.past_window < 1:
            raise ValueError("past_window must be at least 1")
        if self.future_window < 1:
            raise ValueError("future_window must be at least 1")
        if self.bev_frame_count < 0:
            raise ValueError("bev_frame_count must be non-negative")
        if self.alignment_frame_count < 0:
            raise ValueError("alignment_frame_count must be non-negative")
        if self.alignment_tolerance_voxels < 0:
            raise ValueError("alignment_tolerance_voxels must be non-negative")
        if not self.thresholds:
            raise ValueError("thresholds must contain at least one value")
        if not self.fusion_profiles:
            raise ValueError("fusion_profiles must contain at least one profile")
        if self.class_names is None:
            object.__setattr__(self, "class_names", CARLA_CLASS_NAMES)


@dataclass(frozen=True)
class FuturePseudoEvalArtifacts:
    """Paths written by the future pseudo-label Phase 2 evaluation."""

    baseline_iou: Path
    temporal_iou: Path
    eval_summary: Path
    shadow_records: Path
    shadow_clusters: Path
    alignment_summary: Path
    ablation_summary: Path
    bev_images: tuple[Path, ...]
    alignment_images: tuple[Path, ...]


SCENARIO_MEMORY_SLICE_NAMES = (
    "vehicle_change",
    "sparse_depth",
    "turning",
    "dense_traffic",
    "high_disagreement",
)


@dataclass(frozen=True)
class ScenarioMemoryEvalConfig:
    """Configuration for heuristic, non-exclusive scenario-memory evaluation slices.

    Slice selection is scene-derived from occupancy grids and frame metadata. The
    reported IoU rows still follow ``class_ids`` so callers can restrict metrics
    without changing which scene anchors are eligible for scenario slices.
    """

    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "validation"
    max_scan_frames: int = 2000
    anchors_per_slice: int = 50
    output_dir: str | Path = "outputs/phase2/scenario_memory"
    min_vehicle_union: int = 1
    sparse_depth_quantile: float = 0.25
    turn_steer_threshold: float = 0.15
    turn_yaw_delta_threshold_degrees: float = 5.0
    dense_traffic_min_nearby: int = 10
    occupancy_threshold: float = 0.4
    disagreement_threshold: float = 0.05
    bev_frame_count_per_slice: int = 2
    past_window: int = 5
    future_window: int = 5
    fusion_weights: tuple[float, ...] = DEFAULT_STRICT_FUSION_WEIGHTS
    preprocessing: CarlaDataPreprocessingConfig | None = None
    grid_spec: OccupancyGridSpec | None = None
    class_ids: tuple[int, ...] = DEFAULT_CLASS_IDS
    class_names: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        if self.max_scan_frames < 1:
            raise ValueError("max_scan_frames must be at least 1")
        if self.anchors_per_slice < 1:
            raise ValueError("anchors_per_slice must be at least 1")
        if self.min_vehicle_union < 1:
            raise ValueError("min_vehicle_union must be at least 1")
        if not 0.0 <= self.sparse_depth_quantile <= 1.0:
            raise ValueError("sparse_depth_quantile must be between 0 and 1")
        if self.bev_frame_count_per_slice < 0:
            raise ValueError("bev_frame_count_per_slice must be non-negative")
        if self.past_window < 1:
            raise ValueError("past_window must be at least 1")
        if self.future_window < 1:
            raise ValueError("future_window must be at least 1")
        if not self.fusion_weights:
            raise ValueError("fusion_weights must contain at least one weight")
        if self.class_names is None:
            object.__setattr__(self, "class_names", CARLA_CLASS_NAMES)


@dataclass(frozen=True)
class ScenarioMemoryEvalArtifacts:
    """Paths written by scenario-sliced temporal-memory evaluation."""

    eval_summary: Path
    by_slice_iou: Path
    selected_anchors: Path
    bev_images: tuple[Path, ...]


@dataclass(frozen=True)
class ScenarioMemoryAnchor:
    """One anchor frame selected for one or more heuristic scenario slices."""

    anchor_index: int
    run_id: str | None
    frame: int | None
    slice_names: tuple[str, ...]
    depth_coverage: float
    abs_steer: float
    yaw_delta_degrees: float
    nearby_vehicles_50m: int
    total_npc_vehicles: int
    disagreement_rate: float
    scene_vehicle_union: int
    new_vehicle_voxel_count: int

    def to_json_record(self, *, slice_name: str) -> dict[str, Any]:
        return {
            "slice": slice_name,
            "anchor_index": self.anchor_index,
            "run_id": self.run_id,
            "frame": self.frame,
            "all_slice_labels": list(self.slice_names),
            "depth_coverage": self.depth_coverage,
            "abs_steer": self.abs_steer,
            "yaw_delta_degrees": self.yaw_delta_degrees,
            "nearby_vehicles_50m": self.nearby_vehicles_50m,
            "total_npc_vehicles": self.total_npc_vehicles,
            "disagreement_rate": self.disagreement_rate,
            "scene_vehicle_union": self.scene_vehicle_union,
            "new_vehicle_voxel_count": self.new_vehicle_voxel_count,
        }


@dataclass(frozen=True)
class _ScenarioMemoryCandidate:
    anchor: ScenarioMemoryAnchor
    baseline: SemanticOccupancyGrid
    temporal: SemanticOccupancyGrid
    target: SemanticOccupancyGrid
    baseline_iou: Mapping[int, ClassIoU]
    temporal_iou: Mapping[int, ClassIoU]


@dataclass(frozen=True)
class FrameSkipExample:
    """Compact example of a skipped frame error for notebook diagnostics."""

    frame_index: int
    run_id: str | None
    frame: int | None
    error_type: str
    message: str


def stream_huggingface_samples(dataset_name: str, split: str) -> Iterable[Mapping[str, Any]]:
    """Stream raw dataset samples from Hugging Face."""

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    return iter(dataset)


def ego_pose_from_sample(sample: Mapping[str, Any]) -> np.ndarray:
    """Build a vehicle-to-world ego pose matrix from one CARLA sample."""

    return carla_pose_to_matrix(
        x=float(sample.get("location_x", 0.0) or 0.0),
        y=float(sample.get("location_y", 0.0) or 0.0),
        z=float(sample.get("location_z", 0.0) or 0.0),
        pitch=float(sample.get("rotation_pitch", 0.0) or 0.0),
        yaw=float(sample.get("rotation_yaw", 0.0) or 0.0),
        roll=float(sample.get("rotation_roll", 0.0) or 0.0),
        degrees=True,
    )


def fused_to_semantic_grid(
    fused: FusedOccupancyGrid,
    *,
    occupancy_threshold: float,
) -> SemanticOccupancyGrid:
    """Convert a fused occupancy score grid to a semantic grid for visualization."""

    occupied = fused.occupancy_score >= occupancy_threshold
    semantic = np.where(occupied, fused.semantic, UNKNOWN_CLASS).astype(np.int16)
    counts = occupied.astype(np.uint16)
    return SemanticOccupancyGrid(fused.spec, occupied, semantic, counts)


def _processed_sample_from_raw(
    sample: Mapping[str, Any],
    *,
    preprocessing: CarlaDataPreprocessingConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    dataset = CarlaMultimodalDataset([sample], preprocessing=preprocessing)
    return dataset[0], dataset.extrinsics


def _grid_from_processed_sample(
    processed: Mapping[str, Any],
    *,
    extrinsics: np.ndarray,
    fov_degrees: float,
    spec: OccupancyGridSpec,
) -> SemanticOccupancyGrid:
    rgb = processed["rgb"]
    depth = processed["depth"].detach().cpu().numpy()
    segmentation = processed["segmentation"].detach().cpu().numpy()
    height = int(rgb.shape[1])
    width = int(rgb.shape[2])
    intrinsics = estimate_front_camera_intrinsics(width, height, fov_degrees)
    return build_semantic_occupancy_grid(
        depth,
        segmentation,
        intrinsics,
        extrinsics,
        spec=spec,
    )


def _topdown_semantic_image(grid: SemanticOccupancyGrid) -> np.ndarray:
    image = np.full(grid.spec.shape[:2], UNKNOWN_CLASS, dtype=np.int16)
    occupied_xy = np.any(grid.occupied, axis=2)
    for x_idx, y_idx in np.argwhere(occupied_xy):
        z_indices = np.flatnonzero(grid.occupied[x_idx, y_idx])
        if z_indices.size:
            image[x_idx, y_idx] = grid.semantic[x_idx, y_idx, z_indices[-1]]
    return image.T


def _crop_panels_to_occupied(
    panels: list[np.ndarray],
    *,
    padding: int = 8,
) -> list[np.ndarray]:
    occupied = np.zeros(panels[0].shape, dtype=bool)
    for panel in panels:
        occupied |= panel != UNKNOWN_CLASS
    rows, cols = np.nonzero(occupied)
    if rows.size == 0:
        return panels

    row_start = max(int(rows.min()) - padding, 0)
    row_end = min(int(rows.max()) + padding + 1, panels[0].shape[0])
    col_start = max(int(cols.min()) - padding, 0)
    col_end = min(int(cols.max()) + padding + 1, panels[0].shape[1])
    return [panel[row_start:row_end, col_start:col_end] for panel in panels]


def save_bev_comparison(
    *,
    baseline: SemanticOccupancyGrid,
    temporal: SemanticOccupancyGrid,
    target: SemanticOccupancyGrid,
    output_path: str | Path,
    title: str,
    class_names: Mapping[int, str],
    panel_titles: tuple[str, str, str] = ("Baseline", "Temporal fusion", "Target proxy"),
) -> Path:
    """Save baseline, temporal, and target-proxy BEV grids side by side."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise ImportError("matplotlib is required for BEV comparison output") from exc

    panel_arrays = [
        _topdown_semantic_image(baseline),
        _topdown_semantic_image(temporal),
        _topdown_semantic_image(target),
    ]
    panel_arrays = _crop_panels_to_occupied(panel_arrays)
    colors = BEV_CLASS_COLORS
    display_names = BEV_CLASS_NAMES | dict(class_names)
    observed_classes = {int(value) for panel in panel_arrays for value in np.unique(panel)}
    classes = sorted(set(colors) | set(class_names) | observed_classes)
    class_to_index = {class_id: index for index, class_id in enumerate(classes)}
    cmap = mcolors.ListedColormap([colors.get(class_id, "#4c78a8") for class_id in classes])
    panels = list(zip(panel_titles, panel_arrays))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (panel_title, panel) in zip(axes, panels):
        indexed = np.vectorize(class_to_index.__getitem__)(panel)
        ax.imshow(indexed, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(panel_title)
        ax.axis("off")

    handles = [
        mpatches.Patch(
            color=colors.get(class_id, "#4c78a8"),
            label=display_names.get(class_id, f"class {class_id}"),
        )
        for class_id in classes
        if class_id != UNKNOWN_CLASS
    ]
    fig.suptitle(title)
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _semantic_grid_from_transformed_frames(
    frames: Sequence[OccupancyFrame],
    *,
    target_ego_pose: np.ndarray,
    spec: OccupancyGridSpec,
) -> SemanticOccupancyGrid:
    occupied = np.zeros(spec.shape, dtype=bool)
    count_accumulator = np.zeros(spec.shape, dtype=np.uint32)
    semantic = np.full(spec.shape, UNKNOWN_CLASS, dtype=np.int16)
    semantic_votes: dict[tuple[int, int], float] = {}

    for frame in frames:
        if frame.grid.spec != spec:
            raise ValueError("all occupancy frames must use the same grid spec")

        indices, labels, counts = transform_grid_to_ego_frame(
            frame.grid,
            source_ego_pose=frame.ego_pose,
            target_ego_pose=target_ego_pose,
        )
        if indices.size == 0:
            continue

        linear = np.ravel_multi_index((indices[:, 0], indices[:, 1], indices[:, 2]), spec.shape)
        unique_linear, inverse = np.unique(linear, return_inverse=True)
        occupied.flat[unique_linear] = True
        count_accumulator.flat[unique_linear] += np.bincount(
            inverse,
            weights=np.maximum(counts, 1.0),
            minlength=unique_linear.size,
        ).astype(np.uint32)

        for linear_index, label, count in zip(linear, labels, counts):
            semantic_votes[(int(linear_index), int(label))] = semantic_votes.get(
                (int(linear_index), int(label)),
                0.0,
            ) + max(float(count), 1.0)

    best_vote_by_voxel: dict[int, tuple[float, int]] = {}
    for (linear_index, label), vote in semantic_votes.items():
        best_vote, best_label = best_vote_by_voxel.get(linear_index, (-1.0, UNKNOWN_CLASS))
        if vote > best_vote or (np.isclose(vote, best_vote) and label < best_label):
            best_vote_by_voxel[linear_index] = (vote, label)

    for linear_index, (_, label) in best_vote_by_voxel.items():
        semantic.flat[linear_index] = label

    clipped_counts = np.clip(count_accumulator, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return SemanticOccupancyGrid(spec, occupied, semantic, clipped_counts)


def union_grids_in_ego_frame(
    frames: Sequence[OccupancyFrame],
    *,
    target_ego_pose: np.ndarray,
    spec: OccupancyGridSpec,
) -> SemanticOccupancyGrid:
    """Union occupancy frames into a target ego frame using the transform stack."""

    return _semantic_grid_from_transformed_frames(
        frames,
        target_ego_pose=target_ego_pose,
        spec=spec,
    )


def _occupied_overlap_ratio(first: SemanticOccupancyGrid, second: SemanticOccupancyGrid) -> float:
    union = first.occupied | second.occupied
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 1.0
    intersection = first.occupied & second.occupied
    return float(np.count_nonzero(intersection) / union_count)


def _dilate_occupied(occupied: np.ndarray, *, tolerance_voxels: int) -> np.ndarray:
    if tolerance_voxels <= 0:
        return occupied.copy()

    dilated = np.zeros_like(occupied, dtype=bool)
    x_size, y_size, z_size = occupied.shape
    for dx in range(-tolerance_voxels, tolerance_voxels + 1):
        for dy in range(-tolerance_voxels, tolerance_voxels + 1):
            if dx * dx + dy * dy > tolerance_voxels * tolerance_voxels:
                continue

            source_x_start = max(0, -dx)
            source_x_end = min(x_size, x_size - dx)
            source_y_start = max(0, -dy)
            source_y_end = min(y_size, y_size - dy)
            target_x_start = max(0, dx)
            target_x_end = min(x_size, x_size + dx)
            target_y_start = max(0, dy)
            target_y_end = min(y_size, y_size + dy)

            dilated[
                target_x_start:target_x_end,
                target_y_start:target_y_end,
                :z_size,
            ] |= occupied[
                source_x_start:source_x_end,
                source_y_start:source_y_end,
                :z_size,
            ]
    return dilated


def _occupied_near_overlap_ratio(
    first: SemanticOccupancyGrid,
    second: SemanticOccupancyGrid,
    *,
    tolerance_voxels: int,
) -> float:
    union = first.occupied | second.occupied
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 1.0

    first_near_second = first.occupied & _dilate_occupied(
        second.occupied,
        tolerance_voxels=tolerance_voxels,
    )
    second_near_first = second.occupied & _dilate_occupied(
        first.occupied,
        tolerance_voxels=tolerance_voxels,
    )
    near_overlap = first_near_second | second_near_first
    return float(np.count_nonzero(near_overlap) / union_count)


def _anchor_indices(
    *,
    frame_count: int,
    anchor_count: int,
    past_window: int,
    future_window: int,
) -> tuple[int, ...]:
    first_anchor = past_window - 1
    last_anchor_exclusive = frame_count - future_window + 1
    if last_anchor_exclusive <= first_anchor:
        return ()
    return tuple(range(first_anchor, min(last_anchor_exclusive, first_anchor + anchor_count)))


def _all_context_valid_anchor_indices(
    *,
    frame_count: int,
    past_window: int,
    future_window: int,
) -> tuple[int, ...]:
    """Return every anchor with enough past and future context in the scanned buffer."""

    return _anchor_indices(
        frame_count=frame_count,
        anchor_count=frame_count,
        past_window=past_window,
        future_window=future_window,
    )


def _collect_occupancy_frames(
    *,
    config: FuturePseudoEvalConfig,
    raw_samples: Iterable[Mapping[str, Any]] | None,
    max_raw_frames: int | None = None,
) -> tuple[list[OccupancyFrame], dict[str, Any]]:
    preprocessing = config.preprocessing or build_occupancy_preprocessing_config()
    spec = config.grid_spec or OccupancyGridSpec()
    samples = raw_samples
    if samples is None:
        samples = stream_huggingface_samples(config.dataset_name, config.split)

    requested_frames = (
        max_raw_frames
        if max_raw_frames is not None
        else config.anchor_count + config.past_window - 1 + config.future_window
    )
    frames: list[OccupancyFrame] = []
    skipped_errors: dict[str, int] = {}
    skipped_examples: list[FrameSkipExample] = []

    for raw_index, sample in enumerate(islice(samples, requested_frames)):
        try:
            processed, extrinsics = _processed_sample_from_raw(sample, preprocessing=preprocessing)
            grid = _grid_from_processed_sample(
                processed,
                extrinsics=extrinsics,
                fov_degrees=preprocessing.fov_degrees,
                spec=spec,
            )
            metadata = dict(processed["metadata"])
            metadata["index"] = raw_index
            frames.append(OccupancyFrame(grid, ego_pose_from_sample(sample), metadata=metadata))
        except Exception as exc:
            error_name = type(exc).__name__
            skipped_errors[error_name] = skipped_errors.get(error_name, 0) + 1
            if len(skipped_examples) < 5:
                skipped_examples.append(
                    FrameSkipExample(
                        frame_index=raw_index,
                        run_id=None if sample is None else sample.get("run_id"),
                        frame=None if sample is None else sample.get("frame"),
                        error_type=error_name,
                        message=str(exc),
                    )
                )

    summary = {
        "requested_raw_frame_count": requested_frames,
        "processed_raw_frame_count": len(frames),
        "skipped_frame_count": sum(skipped_errors.values()),
        "skipped_errors": skipped_errors,
        "skipped_examples": [example.__dict__ for example in skipped_examples],
    }
    return frames, summary


def _class_row(summary: Mapping[int, ClassIoU], class_id: int) -> dict[str, Any]:
    value = summary[class_id]
    return {
        "iou": value.iou,
        "intersection": value.intersection,
        "union": value.union,
    }


def _ablation_row_score(row: Mapping[str, Any]) -> tuple[float, float]:
    vehicle = row["classes"].get("10", {})
    road = row["classes"].get("7", {})
    vehicle_score = -1.0 if int(vehicle.get("union", 0)) == 0 else float(vehicle.get("iou", 0.0))
    road_score = -1.0 if int(road.get("union", 0)) == 0 else float(road.get("iou", 0.0))
    return vehicle_score, road_score


def _metadata_float(metadata: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    if metadata is None:
        return default
    value = metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_int(metadata: Mapping[str, Any] | None, key: str, default: int = 0) -> int:
    if metadata is None:
        return default
    value = metadata.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metadata_depth_coverage(
    metadata: Mapping[str, Any] | None,
    grid: SemanticOccupancyGrid,
) -> float:
    if metadata is not None:
        depth_summary = metadata.get("depth_summary")
        if isinstance(depth_summary, Mapping):
            for key in ("coverage", "valid_fraction", "valid_ratio", "observed_fraction"):
                if key in depth_summary:
                    try:
                        return float(depth_summary[key])
                    except (TypeError, ValueError):
                        break
    return float(np.count_nonzero(grid.occupied) / grid.occupied.size)


def _yaw_delta_degrees(frames: Sequence[OccupancyFrame], anchor_index: int) -> float:
    if anchor_index <= 0:
        return 0.0
    current = frames[anchor_index].metadata or {}
    previous = frames[anchor_index - 1].metadata or {}
    current_yaw = _metadata_float(current, "rotation_yaw", 0.0)
    previous_yaw = _metadata_float(previous, "rotation_yaw", current_yaw)
    delta = (current_yaw - previous_yaw + 180.0) % 360.0 - 180.0
    return abs(float(delta))


def _vehicle_masks(
    baseline: SemanticOccupancyGrid,
    target: SemanticOccupancyGrid,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_vehicle = baseline.occupied & (baseline.semantic == VEHICLE_CLASS_ID)
    target_vehicle = target.occupied & (target.semantic == VEHICLE_CLASS_ID)
    return baseline_vehicle, target_vehicle


def _scenario_iou_payload(
    baseline_summary: Mapping[int, ClassIoU],
    temporal_summary: Mapping[int, ClassIoU],
    *,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for class_id in sorted(baseline_summary):
        baseline = baseline_summary[class_id]
        temporal = temporal_summary[class_id]
        informative = baseline.union > 0 or temporal.union > 0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "baseline_iou": baseline.iou,
                "temporal_iou": temporal.iou,
                "delta": temporal.iou - baseline.iou,
                "baseline_intersection": baseline.intersection,
                "baseline_union": baseline.union,
                "temporal_intersection": temporal.intersection,
                "temporal_union": temporal.union,
                "informative": informative,
            }
        )
    return {"classes": rows}


def score_scenario_memory_candidates(
    frames: Sequence[OccupancyFrame],
    *,
    config: ScenarioMemoryEvalConfig | None = None,
) -> tuple[list[_ScenarioMemoryCandidate], dict[str, Any]]:
    """Score all context-valid anchors and assign heuristic scenario labels."""

    eval_config = config or ScenarioMemoryEvalConfig()
    if not frames:
        return [], {"candidate_count": 0, "sparse_depth_threshold": None}

    spec = eval_config.grid_spec or frames[0].grid.spec
    anchors = _all_context_valid_anchor_indices(
        frame_count=len(frames),
        past_window=eval_config.past_window,
        future_window=eval_config.future_window,
    )
    if not anchors:
        return [], {"candidate_count": 0, "sparse_depth_threshold": None}

    candidate_inputs: list[dict[str, Any]] = []
    for anchor_index in anchors:
        anchor = frames[anchor_index]
        past_frames = frames[anchor_index - eval_config.past_window + 1 : anchor_index + 1]
        future_frames = frames[anchor_index : anchor_index + eval_config.future_window]
        target = union_grids_in_ego_frame(
            future_frames,
            target_ego_pose=anchor.ego_pose,
            spec=spec,
        )
        fused = fuse_occupancy_frames(
            tuple(reversed(past_frames)), weights=eval_config.fusion_weights
        )
        temporal = fused_to_semantic_grid(
            fused,
            occupancy_threshold=eval_config.occupancy_threshold,
        )
        baseline_iou = compute_semantic_iou(
            anchor.grid,
            target,
            class_ids=eval_config.class_ids,
            occupancy_threshold=eval_config.occupancy_threshold,
        )
        temporal_iou = compute_semantic_iou(
            fused,
            target,
            class_ids=eval_config.class_ids,
            occupancy_threshold=eval_config.occupancy_threshold,
        )
        baseline_vehicle, target_vehicle = _vehicle_masks(anchor.grid, target)
        scene_vehicle_union = int(np.count_nonzero(baseline_vehicle | target_vehicle))
        new_vehicle_voxels = int(np.count_nonzero(target_vehicle & ~baseline_vehicle))
        metadata = anchor.metadata or {}
        coverage = _metadata_depth_coverage(metadata, anchor.grid)
        candidate_inputs.append(
            {
                "anchor_index": anchor_index,
                "baseline": anchor.grid,
                "temporal": temporal,
                "target": target,
                "baseline_iou": baseline_iou,
                "temporal_iou": temporal_iou,
                "depth_coverage": coverage,
                "scene_vehicle_union": scene_vehicle_union,
                "new_vehicle_voxels": new_vehicle_voxels,
                "abs_steer": abs(_metadata_float(metadata, "steer", 0.0)),
                "yaw_delta_degrees": _yaw_delta_degrees(frames, anchor_index),
                "nearby_vehicles_50m": _metadata_int(metadata, "nearby_vehicles_50m", 0),
                "total_npc_vehicles": _metadata_int(metadata, "total_npc_vehicles", 0),
                "disagreement_rate": occupancy_disagreement_rate(
                    anchor.grid,
                    fused,
                    occupancy_threshold=eval_config.occupancy_threshold,
                ),
                "metadata": metadata,
            }
        )

    sparse_threshold = float(
        np.quantile(
            [candidate["depth_coverage"] for candidate in candidate_inputs],
            eval_config.sparse_depth_quantile,
        )
    )
    candidates: list[_ScenarioMemoryCandidate] = []
    for candidate in candidate_inputs:
        slice_names: list[str] = []
        if (
            candidate["scene_vehicle_union"] >= eval_config.min_vehicle_union
            and candidate["new_vehicle_voxels"] >= eval_config.min_vehicle_union
        ):
            slice_names.append("vehicle_change")
        if candidate["depth_coverage"] <= sparse_threshold:
            slice_names.append("sparse_depth")
        if (
            candidate["abs_steer"] >= eval_config.turn_steer_threshold
            or candidate["yaw_delta_degrees"] >= eval_config.turn_yaw_delta_threshold_degrees
        ):
            slice_names.append("turning")
        if (
            candidate["nearby_vehicles_50m"] >= eval_config.dense_traffic_min_nearby
            or candidate["total_npc_vehicles"] >= eval_config.dense_traffic_min_nearby
        ):
            slice_names.append("dense_traffic")
        if candidate["disagreement_rate"] > eval_config.disagreement_threshold:
            slice_names.append("high_disagreement")
        if not slice_names:
            continue

        metadata = candidate["metadata"]
        candidates.append(
            _ScenarioMemoryCandidate(
                anchor=ScenarioMemoryAnchor(
                    anchor_index=int(candidate["anchor_index"]),
                    run_id=metadata.get("run_id"),
                    frame=metadata.get("frame"),
                    slice_names=tuple(slice_names),
                    depth_coverage=float(candidate["depth_coverage"]),
                    abs_steer=float(candidate["abs_steer"]),
                    yaw_delta_degrees=float(candidate["yaw_delta_degrees"]),
                    nearby_vehicles_50m=int(candidate["nearby_vehicles_50m"]),
                    total_npc_vehicles=int(candidate["total_npc_vehicles"]),
                    disagreement_rate=float(candidate["disagreement_rate"]),
                    scene_vehicle_union=int(candidate["scene_vehicle_union"]),
                    new_vehicle_voxel_count=int(candidate["new_vehicle_voxels"]),
                ),
                baseline=candidate["baseline"],
                temporal=candidate["temporal"],
                target=candidate["target"],
                baseline_iou=candidate["baseline_iou"],
                temporal_iou=candidate["temporal_iou"],
            )
        )

    return candidates, {
        "candidate_count": len(candidate_inputs),
        "labeled_candidate_count": len(candidates),
        "sparse_depth_threshold": sparse_threshold,
    }


def _select_candidates_by_slice(
    candidates: Sequence[_ScenarioMemoryCandidate],
    *,
    anchors_per_slice: int,
) -> dict[str, list[_ScenarioMemoryCandidate]]:
    selected: dict[str, list[_ScenarioMemoryCandidate]] = {
        slice_name: [] for slice_name in SCENARIO_MEMORY_SLICE_NAMES
    }
    for candidate in candidates:
        for slice_name in candidate.anchor.slice_names:
            if len(selected[slice_name]) < anchors_per_slice:
                selected[slice_name].append(candidate)
    return selected


def _write_selected_anchors_jsonl(
    path: Path,
    selected_by_slice: Mapping[str, Sequence[_ScenarioMemoryCandidate]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for slice_name, candidates in selected_by_slice.items():
            for candidate in candidates:
                handle.write(
                    json.dumps(candidate.anchor.to_json_record(slice_name=slice_name)) + "\n"
                )
    return path


def run_scenario_memory_evaluation(
    *,
    config: ScenarioMemoryEvalConfig | None = None,
    raw_samples: Iterable[Mapping[str, Any]] | None = None,
    frames: Sequence[OccupancyFrame] | None = None,
) -> ScenarioMemoryEvalArtifacts:
    """Evaluate future-pseudo IoU on anchors where temporal memory should matter."""

    eval_config = config or ScenarioMemoryEvalConfig()
    class_names = eval_config.class_names or CARLA_CLASS_NAMES
    output_dir = Path(eval_config.output_dir)
    bev_dir = output_dir / "scenario_memory_bev"
    collection_summary: dict[str, Any] = {}

    frame_sequence: Sequence[OccupancyFrame]
    if frames is None:
        collection_config = FuturePseudoEvalConfig(
            dataset_name=eval_config.dataset_name,
            split=eval_config.split,
            anchor_count=eval_config.max_scan_frames,
            output_dir=eval_config.output_dir,
            occupancy_threshold=eval_config.occupancy_threshold,
            disagreement_threshold=eval_config.disagreement_threshold,
            past_window=eval_config.past_window,
            future_window=eval_config.future_window,
            preprocessing=eval_config.preprocessing,
            grid_spec=eval_config.grid_spec,
            class_ids=eval_config.class_ids,
            class_names=class_names,
        )
        loaded_frames, collection_summary = _collect_occupancy_frames(
            config=collection_config,
            raw_samples=raw_samples,
            max_raw_frames=eval_config.max_scan_frames,
        )
        frame_sequence = loaded_frames
    else:
        frame_sequence = frames
        collection_summary = {
            "requested_raw_frame_count": len(frame_sequence),
            "processed_raw_frame_count": len(frame_sequence),
            "skipped_frame_count": 0,
            "skipped_errors": {},
            "skipped_examples": [],
        }

    candidates, candidate_summary = score_scenario_memory_candidates(
        frame_sequence,
        config=eval_config,
    )
    selected_by_slice = _select_candidates_by_slice(
        candidates,
        anchors_per_slice=eval_config.anchors_per_slice,
    )

    by_slice_payload: dict[str, Any] = {
        "evaluation_name": "scenario_memory_future_pseudo_iou",
        "target_proxy": (
            "future union of projected-depth semantic occupancy grids transformed into "
            "the anchor ego frame; not dense simulator 3D ground truth"
        ),
        "class_ids": {
            str(class_id): class_names.get(class_id, str(class_id))
            for class_id in eval_config.class_ids
        },
        "slices": {},
    }
    bev_images: list[Path] = []
    visualization_errors: dict[str, int] = {}

    for slice_name, selected in selected_by_slice.items():
        baseline_results = [candidate.baseline_iou for candidate in selected]
        temporal_results = [candidate.temporal_iou for candidate in selected]
        baseline_summary = summarize_iou(baseline_results)
        temporal_summary = summarize_iou(temporal_results)
        by_slice_payload["slices"][slice_name] = {
            "selected_anchor_count": len(selected),
            **_scenario_iou_payload(
                baseline_summary,
                temporal_summary,
                class_names=class_names,
            ),
        }

        saved_for_slice = 0
        for candidate in selected:
            if saved_for_slice >= eval_config.bev_frame_count_per_slice:
                break
            frame_label = candidate.anchor.frame or candidate.anchor.anchor_index
            try:
                bev_images.append(
                    save_bev_comparison(
                        baseline=candidate.baseline,
                        temporal=candidate.temporal,
                        target=candidate.target,
                        output_path=bev_dir
                        / slice_name
                        / f"{slice_name}_{candidate.anchor.anchor_index:05d}_{frame_label}.png",
                        title=f"{slice_name}: frame {frame_label} future pseudo-label occupancy",
                        class_names=class_names,
                        panel_titles=("Baseline t", "Past temporal fusion", "Future pseudo-label"),
                    )
                )
                saved_for_slice += 1
            except Exception as exc:
                error_name = type(exc).__name__
                visualization_errors[error_name] = visualization_errors.get(error_name, 0) + 1

    selected_path = _write_selected_anchors_jsonl(
        output_dir / "scenario_memory_selected_anchors.jsonl",
        selected_by_slice,
    )
    by_slice_path = _write_json(output_dir / "scenario_memory_by_slice_iou.json", by_slice_payload)
    summary_path = _write_json(
        output_dir / "scenario_memory_eval_summary.json",
        {
            "evaluation_name": "scenario_memory_future_pseudo_iou",
            "dataset_name": eval_config.dataset_name,
            "split": eval_config.split,
            "max_scan_frames": eval_config.max_scan_frames,
            "anchors_per_slice": eval_config.anchors_per_slice,
            "past_window": eval_config.past_window,
            "future_window": eval_config.future_window,
            "occupancy_threshold": eval_config.occupancy_threshold,
            "disagreement_threshold": eval_config.disagreement_threshold,
            "fusion_weights": list(eval_config.fusion_weights),
            "slice_names": list(SCENARIO_MEMORY_SLICE_NAMES),
            "selected_anchor_counts": {
                slice_name: len(selected) for slice_name, selected in selected_by_slice.items()
            },
            "selection_thresholds": {
                "min_scene_vehicle_union": eval_config.min_vehicle_union,
                "sparse_depth_quantile": eval_config.sparse_depth_quantile,
                "sparse_depth_threshold": candidate_summary["sparse_depth_threshold"],
                "turn_steer_threshold": eval_config.turn_steer_threshold,
                "turn_yaw_delta_threshold_degrees": eval_config.turn_yaw_delta_threshold_degrees,
                "dense_traffic_min_nearby": eval_config.dense_traffic_min_nearby,
            },
            "visualization_errors": visualization_errors,
            "bev_images": [str(path) for path in bev_images],
            **candidate_summary,
            **collection_summary,
        },
    )

    return ScenarioMemoryEvalArtifacts(
        eval_summary=summary_path,
        by_slice_iou=by_slice_path,
        selected_anchors=selected_path,
        bev_images=tuple(bev_images),
    )


def run_future_pseudo_label_evaluation(
    *,
    config: FuturePseudoEvalConfig | None = None,
    raw_samples: Iterable[Mapping[str, Any]] | None = None,
) -> FuturePseudoEvalArtifacts:
    """Evaluate baseline and temporal fusion against a future multi-frame pseudo-label."""

    eval_config = config or FuturePseudoEvalConfig()
    spec = eval_config.grid_spec or OccupancyGridSpec()
    class_names = eval_config.class_names or CARLA_CLASS_NAMES
    output_dir = Path(eval_config.output_dir)
    bev_dir = output_dir / "future_pseudo_bev"
    alignment_dir = output_dir / "alignment"

    frames, collection_summary = _collect_occupancy_frames(
        config=eval_config, raw_samples=raw_samples
    )
    anchors = _anchor_indices(
        frame_count=len(frames),
        anchor_count=eval_config.anchor_count,
        past_window=eval_config.past_window,
        future_window=eval_config.future_window,
    )
    if not anchors:
        _write_json(
            output_dir / "future_pseudo_eval_summary.json",
            {
                "evaluation_name": "future_pseudo_label_semantic_occupancy_iou",
                "anchor_count": 0,
                "skipped_anchor_count": max(eval_config.anchor_count, 1),
                "reason": "insufficient processed context for requested past/future windows",
                **collection_summary,
            },
        )
        raise RuntimeError("No anchors had enough past and future context for future-pseudo IoU")

    target_by_anchor: dict[int, SemanticOccupancyGrid] = {}
    baseline_iou_results: list[Mapping[int, ClassIoU]] = []
    ablation_iou_results: dict[tuple[str, float], list[Mapping[int, ClassIoU]]] = {
        (profile.name, threshold): []
        for profile in eval_config.fusion_profiles
        for threshold in eval_config.thresholds
    }
    shadow_evaluator = ShadowModeEvaluator(
        disagreement_threshold=eval_config.disagreement_threshold,
        occupancy_threshold=eval_config.occupancy_threshold,
        class_ids=eval_config.class_ids,
        class_names=class_names,
    )

    default_profile = eval_config.fusion_profiles[0]
    bev_images: list[Path] = []
    alignment_images: list[Path] = []
    alignment_records: list[dict[str, Any]] = []
    visualization_errors: dict[str, int] = {}

    for anchor_position, anchor_index in enumerate(anchors):
        anchor = frames[anchor_index]
        past_frames = frames[anchor_index - eval_config.past_window + 1 : anchor_index + 1]
        past_frames_newest_first = tuple(reversed(past_frames))
        future_frames = frames[anchor_index : anchor_index + eval_config.future_window]
        target = union_grids_in_ego_frame(
            future_frames,
            target_ego_pose=anchor.ego_pose,
            spec=spec,
        )
        target_by_anchor[anchor_index] = target

        baseline_iou_results.append(
            compute_semantic_iou(anchor.grid, target, class_ids=eval_config.class_ids)
        )

        for profile in eval_config.fusion_profiles:
            fused = fuse_occupancy_frames(past_frames_newest_first, weights=profile.weights)
            for threshold in eval_config.thresholds:
                ablation_iou_results[(profile.name, threshold)].append(
                    compute_semantic_iou(
                        fused,
                        target,
                        class_ids=eval_config.class_ids,
                        occupancy_threshold=threshold,
                    )
                )

        default_fused = fuse_occupancy_frames(
            past_frames_newest_first,
            weights=default_profile.weights,
        )
        shadow_evaluator.evaluate_frame(
            baseline=anchor.grid,
            improved=default_fused,
            metadata=anchor.metadata or {},
        )

        if len(bev_images) < eval_config.bev_frame_count:
            temporal = fused_to_semantic_grid(
                default_fused,
                occupancy_threshold=eval_config.occupancy_threshold,
            )
            frame_label = (anchor.metadata or {}).get("frame", anchor_index)
            try:
                bev_images.append(
                    save_bev_comparison(
                        baseline=anchor.grid,
                        temporal=temporal,
                        target=target,
                        output_path=bev_dir
                        / f"future_pseudo_bev_{anchor_position:05d}_{frame_label}.png",
                        title=f"Frame {frame_label} future pseudo-label occupancy",
                        class_names=class_names,
                        panel_titles=("Baseline t", "Past temporal fusion", "Future pseudo-label"),
                    )
                )
            except Exception as exc:
                error_name = type(exc).__name__
                visualization_errors[error_name] = visualization_errors.get(error_name, 0) + 1

        if len(alignment_images) < eval_config.alignment_frame_count and anchor_index > 0:
            previous_aligned = union_grids_in_ego_frame(
                (frames[anchor_index - 1],),
                target_ego_pose=anchor.ego_pose,
                spec=spec,
            )
            previous_overlap = _occupied_overlap_ratio(anchor.grid, previous_aligned)
            future_overlap = _occupied_overlap_ratio(anchor.grid, target)
            previous_near_overlap = _occupied_near_overlap_ratio(
                anchor.grid,
                previous_aligned,
                tolerance_voxels=eval_config.alignment_tolerance_voxels,
            )
            future_near_overlap = _occupied_near_overlap_ratio(
                anchor.grid,
                target,
                tolerance_voxels=eval_config.alignment_tolerance_voxels,
            )
            frame_label = (anchor.metadata or {}).get("frame", anchor_index)
            record = {
                "anchor_position": anchor_position,
                "anchor_index": anchor_index,
                "frame": frame_label,
                "previous_overlap_ratio": previous_overlap,
                "future_overlap_ratio": future_overlap,
                "previous_near_overlap_ratio": previous_near_overlap,
                "future_near_overlap_ratio": future_near_overlap,
            }
            alignment_records.append(record)
            try:
                alignment_images.append(
                    save_bev_comparison(
                        baseline=anchor.grid,
                        temporal=previous_aligned,
                        target=target,
                        output_path=alignment_dir
                        / f"alignment_frame_{anchor_position:05d}_{frame_label}.png",
                        title=f"Frame {frame_label} temporal alignment diagnostic",
                        class_names=class_names,
                        panel_titles=("Current t", "Previous aligned to t", "Future target in t"),
                    )
                )
                record["image_path"] = str(alignment_images[-1])
            except Exception as exc:
                error_name = type(exc).__name__
                visualization_errors[error_name] = visualization_errors.get(error_name, 0) + 1

    baseline_summary = summarize_iou(baseline_iou_results)
    ablation_rows: list[dict[str, Any]] = []
    for (profile_name, threshold), results in ablation_iou_results.items():
        summary = summarize_iou(results)
        row = {
            "profile": profile_name,
            "occupancy_threshold": threshold,
            "classes": {
                str(class_id): {
                    "class_name": class_names.get(class_id, str(class_id)),
                    **_class_row(summary, class_id),
                }
                for class_id in eval_config.class_ids
            },
        }
        ablation_rows.append(row)

    best_row = max(ablation_rows, key=_ablation_row_score)
    best_profile = next(
        profile for profile in eval_config.fusion_profiles if profile.name == best_row["profile"]
    )
    best_threshold = float(best_row["occupancy_threshold"])

    temporal_iou_results: list[Mapping[int, ClassIoU]] = []
    for anchor_index in anchors:
        past_frames = frames[anchor_index - eval_config.past_window + 1 : anchor_index + 1]
        fused = fuse_occupancy_frames(tuple(reversed(past_frames)), weights=best_profile.weights)
        temporal_iou_results.append(
            compute_semantic_iou(
                fused,
                target_by_anchor[anchor_index],
                class_ids=eval_config.class_ids,
                occupancy_threshold=best_threshold,
            )
        )
    temporal_summary = summarize_iou(temporal_iou_results)

    baseline_path = write_iou_summary_json(
        baseline_summary,
        output_dir / "future_pseudo_baseline_iou_summary.json",
        class_names=class_names,
    )
    temporal_path = write_iou_summary_json(
        temporal_summary,
        output_dir / "future_pseudo_temporal_iou_summary.json",
        class_names=class_names,
    )
    records_path = shadow_evaluator.write_records_jsonl(
        output_dir / "future_pseudo_shadow_records.jsonl"
    )
    clusters_path = shadow_evaluator.write_cluster_summary_json(
        output_dir / "future_pseudo_shadow_clusters.json"
    )
    ablation_path = _write_json(
        output_dir / "fusion_ablation_summary.json",
        {
            "selection_rule": "primary vehicle IoU, secondary road IoU; empty-union classes score -1",
            "selected_profile": best_profile.name,
            "selected_weights": list(best_profile.weights),
            "selected_occupancy_threshold": best_threshold,
            "rows": ablation_rows,
        },
    )
    alignment_path = _write_json(
        output_dir / "alignment_summary.json",
        {
            "diagnostic": "adjacent and future occupancy transformed into anchor ego frame",
            "frame_count": len(alignment_records),
            "near_overlap_tolerance_voxels": eval_config.alignment_tolerance_voxels,
            "mean_previous_overlap_ratio": (
                float(np.mean([record["previous_overlap_ratio"] for record in alignment_records]))
                if alignment_records
                else None
            ),
            "mean_future_overlap_ratio": (
                float(np.mean([record["future_overlap_ratio"] for record in alignment_records]))
                if alignment_records
                else None
            ),
            "mean_previous_near_overlap_ratio": (
                float(
                    np.mean([record["previous_near_overlap_ratio"] for record in alignment_records])
                )
                if alignment_records
                else None
            ),
            "mean_future_near_overlap_ratio": (
                float(
                    np.mean([record["future_near_overlap_ratio"] for record in alignment_records])
                )
                if alignment_records
                else None
            ),
            "records": alignment_records,
            "image_paths": [str(path) for path in alignment_images],
        },
    )
    summary_path = _write_json(
        output_dir / "future_pseudo_eval_summary.json",
        {
            "evaluation_name": "future_pseudo_label_semantic_occupancy_iou",
            "dataset_name": eval_config.dataset_name,
            "split": eval_config.split,
            "requested_anchor_count": eval_config.anchor_count,
            "processed_anchor_count": len(anchors),
            "skipped_anchor_count": max(eval_config.anchor_count - len(anchors), 0),
            "past_window": eval_config.past_window,
            "future_window": eval_config.future_window,
            "selected_profile": best_profile.name,
            "selected_weights": list(best_profile.weights),
            "selected_occupancy_threshold": best_threshold,
            "disagreement_threshold": eval_config.disagreement_threshold,
            "target_proxy": (
                "future union of projected-depth semantic occupancy grids from frames "
                "t through t+future_window-1 transformed into frame t; this is not dense "
                "simulator 3D ground truth"
            ),
            "class_ids": {
                str(class_id): class_names.get(class_id, str(class_id))
                for class_id in eval_config.class_ids
            },
            "visualization_errors": visualization_errors,
            "bev_images": [str(path) for path in bev_images],
            "alignment_images": [str(path) for path in alignment_images],
            **collection_summary,
        },
    )

    return FuturePseudoEvalArtifacts(
        baseline_iou=baseline_path,
        temporal_iou=temporal_path,
        eval_summary=summary_path,
        shadow_records=records_path,
        shadow_clusters=clusters_path,
        alignment_summary=alignment_path,
        ablation_summary=ablation_path,
        bev_images=tuple(bev_images),
        alignment_images=tuple(alignment_images),
    )


def run_bounded_carla_iou_evaluation(
    *,
    config: Phase2CarlaEvalConfig | None = None,
    raw_samples: Iterable[Mapping[str, Any]] | None = None,
) -> Phase2CarlaEvalArtifacts:
    """Run bounded Phase 2 evaluation and write notebook-readable artifacts."""

    eval_config = config or Phase2CarlaEvalConfig()
    preprocessing = eval_config.preprocessing or build_occupancy_preprocessing_config()
    spec = eval_config.grid_spec or OccupancyGridSpec()
    class_names = eval_config.class_names or CARLA_CLASS_NAMES
    output_dir = Path(eval_config.output_dir)
    bev_dir = output_dir / "bev"

    samples = raw_samples
    if samples is None:
        samples = stream_huggingface_samples(eval_config.dataset_name, eval_config.split)

    buffer = TemporalFusionBuffer()
    evaluator = ShadowModeEvaluator(
        disagreement_threshold=eval_config.disagreement_threshold,
        occupancy_threshold=eval_config.occupancy_threshold,
        class_ids=eval_config.class_ids,
        class_names=class_names,
    )
    baseline_iou_results: list[Mapping[int, ClassIoU]] = []
    temporal_iou_results: list[Mapping[int, ClassIoU]] = []
    bev_images: list[Path] = []
    processed_count = 0
    skipped_count = 0
    skipped_errors: dict[str, int] = {}
    skipped_examples: list[FrameSkipExample] = []
    visualization_errors: dict[str, int] = {}

    for frame_index, sample in enumerate(islice(samples, eval_config.max_frames)):
        try:
            processed, extrinsics = _processed_sample_from_raw(sample, preprocessing=preprocessing)
            baseline = _grid_from_processed_sample(
                processed,
                extrinsics=extrinsics,
                fov_degrees=preprocessing.fov_degrees,
                spec=spec,
            )
            target = baseline
            ego_pose = ego_pose_from_sample(sample)
            fused = buffer.add(
                OccupancyFrame(
                    baseline,
                    ego_pose,
                    metadata=dict(processed["metadata"]),
                )
            )
            temporal = fused_to_semantic_grid(
                fused,
                occupancy_threshold=eval_config.occupancy_threshold,
            )

            baseline_iou_results.append(
                compute_semantic_iou(baseline, target, class_ids=eval_config.class_ids)
            )
            temporal_iou_results.append(
                compute_semantic_iou(
                    fused,
                    target,
                    class_ids=eval_config.class_ids,
                    occupancy_threshold=eval_config.occupancy_threshold,
                )
            )
            evaluator.evaluate_frame(
                baseline=baseline,
                improved=fused,
                metadata=dict(processed["metadata"]),
            )
            processed_count += 1

            should_save_bev = len(bev_images) < eval_config.bev_frame_count
            should_save_flagged = evaluator.records[-1].flagged and len(bev_images) < (
                eval_config.bev_frame_count + 2
            )
            if should_save_bev or should_save_flagged:
                frame_label = processed["metadata"].get("frame", frame_index)
                try:
                    bev_images.append(
                        save_bev_comparison(
                            baseline=baseline,
                            temporal=temporal,
                            target=target,
                            output_path=bev_dir / f"bev_frame_{frame_index:05d}_{frame_label}.png",
                            title=f"Frame {frame_label} projected-depth proxy occupancy",
                            class_names=class_names,
                        )
                    )
                except Exception as exc:
                    error_name = type(exc).__name__
                    visualization_errors[error_name] = visualization_errors.get(error_name, 0) + 1
        except Exception as exc:
            skipped_count += 1
            error_name = type(exc).__name__
            skipped_errors[error_name] = skipped_errors.get(error_name, 0) + 1
            if len(skipped_examples) < 5:
                skipped_examples.append(
                    FrameSkipExample(
                        frame_index=frame_index,
                        run_id=None if sample is None else sample.get("run_id"),
                        frame=None if sample is None else sample.get("frame"),
                        error_type=error_name,
                        message=str(exc),
                    )
                )

    if processed_count == 0:
        details = "; ".join(
            f"{example.error_type} at index {example.frame_index} "
            f"(run_id={example.run_id}, frame={example.frame}): {example.message}"
            for example in skipped_examples
        )
        raise RuntimeError(
            "No frames were processed successfully; cannot write IoU summaries. "
            f"Skipped errors: {skipped_errors}. Examples: {details}"
        )

    baseline_summary = summarize_iou(baseline_iou_results)
    temporal_summary = summarize_iou(temporal_iou_results)
    baseline_path = write_iou_summary_json(
        baseline_summary,
        output_dir / "carla_baseline_iou_summary.json",
        class_names=class_names,
    )
    temporal_path = write_iou_summary_json(
        temporal_summary,
        output_dir / "carla_temporal_iou_summary.json",
        class_names=class_names,
    )
    records_path = evaluator.write_records_jsonl(output_dir / "carla_shadow_mode_records.jsonl")
    clusters_path = evaluator.write_cluster_summary_json(
        output_dir / "carla_shadow_mode_clusters.json"
    )
    summary_path = _write_json(
        output_dir / "carla_evaluation_run_summary.json",
        {
            "evaluation_name": "current_frame_proxy_agreement_diagnostic",
            "dataset_name": eval_config.dataset_name,
            "split": eval_config.split,
            "requested_max_frames": eval_config.max_frames,
            "processed_frame_count": processed_count,
            "skipped_frame_count": skipped_count,
            "skipped_errors": skipped_errors,
            "skipped_examples": [example.__dict__ for example in skipped_examples],
            "visualization_errors": visualization_errors,
            "occupancy_threshold": eval_config.occupancy_threshold,
            "disagreement_threshold": eval_config.disagreement_threshold,
            "target_proxy": (
                "current-frame occupancy grid from LiDAR-converted projected depth and "
                "front semantic segmentation; this is a diagnostic agreement check, not "
                "final IoU or dense simulator 3D ground truth"
            ),
            "class_ids": {
                str(class_id): class_names.get(class_id, str(class_id))
                for class_id in eval_config.class_ids
            },
            "free_space_note": (
                "Free space IoU is not reported because this occupancy representation stores "
                "observed occupied voxels and does not raycast explicit free-space labels."
            ),
            "bev_images": [str(path) for path in bev_images],
        },
    )

    return Phase2CarlaEvalArtifacts(
        baseline_iou=baseline_path,
        temporal_iou=temporal_path,
        shadow_records=records_path,
        shadow_clusters=clusters_path,
        run_summary=summary_path,
        bev_images=tuple(bev_images),
    )


def iter_shadow_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL shadow-mode records from an artifact path."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    artifacts = run_bounded_carla_iou_evaluation()
    for field, path in artifacts.__dict__.items():
        print(f"{field}: {path}")


if __name__ == "__main__":
    main()
