"""Bounded streamed CARLA Phase 2 occupancy evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

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


def _validate_config(
    cfg: Any,
    positive: Sequence[str] = (),
    non_negative: Sequence[str] = (),
) -> None:
    for name in positive:
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    for name in non_negative:
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if cfg.class_names is None:
        object.__setattr__(cfg, "class_names", CARLA_CLASS_NAMES)


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
        _validate_config(self, positive=("max_frames",), non_negative=("bev_frame_count",))


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
        _validate_config(
            self,
            positive=("anchor_count", "past_window", "future_window"),
            non_negative=("bev_frame_count", "alignment_frame_count", "alignment_tolerance_voxels"),
        )
        if not self.thresholds:
            raise ValueError("thresholds must contain at least one value")
        if not self.fusion_profiles:
            raise ValueError("fusion_profiles must contain at least one profile")


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
    """Configuration for heuristic, non-exclusive scenario-memory evaluation slices."""

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
        _validate_config(
            self,
            positive=(
                "max_scan_frames",
                "anchors_per_slice",
                "min_vehicle_union",
                "past_window",
                "future_window",
            ),
            non_negative=("bev_frame_count_per_slice",),
        )
        if not 0.0 <= self.sparse_depth_quantile <= 1.0:
            raise ValueError("sparse_depth_quantile must be between 0 and 1")
        if not self.fusion_weights:
            raise ValueError("fusion_weights must contain at least one weight")


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


class _ErrorTracker:
    """Tracks frame processing skip errors and examples across evaluation loops."""

    def __init__(self) -> None:
        self.errors: dict[str, int] = {}
        self.examples: list[FrameSkipExample] = []

    def record(self, exc: Exception, idx: int, sample: Mapping[str, Any] | None) -> None:
        name = type(exc).__name__
        self.errors[name] = self.errors.get(name, 0) + 1
        if len(self.examples) < 5:
            self.examples.append(
                FrameSkipExample(
                    frame_index=idx,
                    run_id=None if sample is None else sample.get("run_id"),
                    frame=None if sample is None else sample.get("frame"),
                    error_type=name,
                    message=str(exc),
                )
            )

    def summary(self, requested: int, processed: int) -> dict[str, Any]:
        return {
            "requested_raw_frame_count": requested,
            "processed_raw_frame_count": processed,
            "skipped_frame_count": sum(self.errors.values()),
            "skipped_errors": self.errors,
            "skipped_examples": [ex.__dict__ for ex in self.examples],
        }

    def raise_if_empty(self, processed: int) -> None:
        if processed == 0:
            details = "; ".join(
                f"{e.error_type} at index {e.frame_index} (run_id={e.run_id}, frame={e.frame}): {e.message}"
                for e in self.examples
            )
            raise RuntimeError(
                f"No frames were processed successfully; cannot write IoU summaries. "
                f"Skipped errors: {self.errors}. Examples: {details}"
            )


def stream_huggingface_samples(dataset_name: str, split: str) -> Iterable[Mapping[str, Any]]:
    """Stream raw dataset samples from Hugging Face."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True, batch_size=1)
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
    return SemanticOccupancyGrid(fused.spec, occupied, semantic, occupied.astype(np.uint16))


def _frame_from_sample(
    sample: Mapping[str, Any],
    *,
    preprocessing: CarlaDataPreprocessingConfig,
    spec: OccupancyGridSpec,
    raw_index: int | None = None,
) -> OccupancyFrame:
    """Build an OccupancyFrame from a raw dataset sample."""
    dataset = CarlaMultimodalDataset([sample], preprocessing=preprocessing)
    processed, extrinsics = dataset[0], dataset.extrinsics
    rgb = processed["rgb"]
    intrinsics = estimate_front_camera_intrinsics(
        int(rgb.shape[2]), int(rgb.shape[1]), preprocessing.fov_degrees
    )
    grid = build_semantic_occupancy_grid(
        processed["depth"].detach().cpu().numpy(),
        processed["segmentation"].detach().cpu().numpy(),
        intrinsics,
        extrinsics,
        spec=spec,
    )
    metadata = dict(processed["metadata"])
    if raw_index is not None:
        metadata["index"] = raw_index
    return OccupancyFrame(grid, ego_pose_from_sample(sample), metadata=metadata)


def _topdown_semantic_image(grid: SemanticOccupancyGrid) -> np.ndarray:
    image = np.full(grid.spec.shape[:2], UNKNOWN_CLASS, dtype=np.int16)
    occupied_xy = np.any(grid.occupied, axis=2)
    for x_idx, y_idx in np.argwhere(occupied_xy):
        z_indices = np.flatnonzero(grid.occupied[x_idx, y_idx])
        if z_indices.size:
            image[x_idx, y_idx] = grid.semantic[x_idx, y_idx, z_indices[-1]]
    return image.T


def _crop_panels_to_occupied(panels: list[np.ndarray], *, padding: int = 8) -> list[np.ndarray]:
    occupied = np.zeros(panels[0].shape, dtype=bool)
    for panel in panels:
        occupied |= panel != UNKNOWN_CLASS
    rows, cols = np.nonzero(occupied)
    if rows.size == 0:
        return panels
    r0, r1 = (
        max(int(rows.min()) - padding, 0),
        min(int(rows.max()) + padding + 1, panels[0].shape[0]),
    )
    c0, c1 = (
        max(int(cols.min()) - padding, 0),
        min(int(cols.max()) + padding + 1, panels[0].shape[1]),
    )
    return [panel[r0:r1, c0:c1] for panel in panels]


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
    except ImportError as exc:
        raise ImportError("matplotlib is required for BEV comparison output") from exc

    panel_arrays = _crop_panels_to_occupied(
        [
            _topdown_semantic_image(baseline),
            _topdown_semantic_image(temporal),
            _topdown_semantic_image(target),
        ]
    )
    colors = BEV_CLASS_COLORS
    display_names = BEV_CLASS_NAMES | dict(class_names)
    observed_classes = {int(v) for p in panel_arrays for v in np.unique(p)}
    classes = sorted(set(colors) | set(class_names) | observed_classes)
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    cmap = mcolors.ListedColormap([colors.get(c, "#4c78a8") for c in classes])

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (panel_title, panel) in zip(axes, zip(panel_titles, panel_arrays)):
        indexed = np.vectorize(class_to_index.__getitem__)(panel)
        ax.imshow(indexed, origin="lower", cmap=cmap, interpolation="nearest")
        ax.set_title(panel_title)
        ax.axis("off")

    handles = [
        mpatches.Patch(color=colors.get(c, "#4c78a8"), label=display_names.get(c, f"class {c}"))
        for c in classes
        if c != UNKNOWN_CLASS
    ]
    fig.suptitle(title)
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _try_save_bev(
    *,
    baseline: SemanticOccupancyGrid,
    temporal: SemanticOccupancyGrid,
    target: SemanticOccupancyGrid,
    output_path: Path,
    title: str,
    class_names: Mapping[int, str],
    panel_titles: tuple[str, str, str] = ("Baseline", "Temporal fusion", "Target proxy"),
    error_tracker: dict[str, int],
    images: list[Path] | None = None,
) -> Path | None:
    """Save BEV comparison and record any visualization errors safely."""
    try:
        path = save_bev_comparison(
            baseline=baseline,
            temporal=temporal,
            target=target,
            output_path=output_path,
            title=title,
            class_names=class_names,
            panel_titles=panel_titles,
        )
        if images is not None:
            images.append(path)
        return path
    except Exception as exc:
        err_name = type(exc).__name__
        error_tracker[err_name] = error_tracker.get(err_name, 0) + 1
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def union_grids_in_ego_frame(
    frames: Sequence[OccupancyFrame],
    *,
    target_ego_pose: np.ndarray,
    spec: OccupancyGridSpec,
) -> SemanticOccupancyGrid:
    """Union occupancy frames into a target ego frame using the transform stack."""
    if not frames:
        return SemanticOccupancyGrid(
            spec,
            np.zeros(spec.shape, dtype=bool),
            np.full(spec.shape, UNKNOWN_CLASS, dtype=np.int16),
            np.zeros(spec.shape, dtype=np.uint16),
        )
    fused = fuse_occupancy_frames(
        frames,
        weights=[1.0] * len(frames),
        target_ego_pose=target_ego_pose,
    )
    return fused_to_semantic_grid(fused, occupancy_threshold=1e-6)


def _occupied_overlap_ratio(first: SemanticOccupancyGrid, second: SemanticOccupancyGrid) -> float:
    union = first.occupied | second.occupied
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 1.0
    return float(np.count_nonzero(first.occupied & second.occupied) / union_count)


def _dilate_occupied(occupied: np.ndarray, *, tolerance_voxels: int) -> np.ndarray:
    if tolerance_voxels <= 0:
        return occupied.copy()

    dilated = np.zeros_like(occupied, dtype=bool)
    x_size, y_size, z_size = occupied.shape
    for dx in range(-tolerance_voxels, tolerance_voxels + 1):
        for dy in range(-tolerance_voxels, tolerance_voxels + 1):
            if dx * dx + dy * dy > tolerance_voxels * tolerance_voxels:
                continue
            sx0, sx1 = max(0, -dx), min(x_size, x_size - dx)
            sy0, sy1 = max(0, -dy), min(y_size, y_size - dy)
            tx0, tx1 = max(0, dx), min(x_size, x_size + dx)
            ty0, ty1 = max(0, dy), min(y_size, y_size + dy)
            dilated[tx0:tx1, ty0:ty1, :z_size] |= occupied[sx0:sx1, sy0:sy1, :z_size]
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
    f_near_s = first.occupied & _dilate_occupied(second.occupied, tolerance_voxels=tolerance_voxels)
    s_near_f = second.occupied & _dilate_occupied(first.occupied, tolerance_voxels=tolerance_voxels)
    return float(np.count_nonzero(f_near_s | s_near_f) / union_count)


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
    config: Any,
    raw_samples: Iterable[Mapping[str, Any]] | None,
    max_raw_frames: int | None = None,
) -> tuple[list[OccupancyFrame], dict[str, Any]]:
    preprocessing = config.preprocessing or build_occupancy_preprocessing_config()
    spec = config.grid_spec or OccupancyGridSpec()
    samples = (
        raw_samples
        if raw_samples is not None
        else stream_huggingface_samples(config.dataset_name, config.split)
    )
    requested = (
        max_raw_frames
        if max_raw_frames is not None
        else config.anchor_count + config.past_window - 1 + config.future_window
    )
    frames: list[OccupancyFrame] = []
    tracker = _ErrorTracker()

    for raw_idx, sample in enumerate(islice(samples, requested)):
        try:
            frames.append(
                _frame_from_sample(
                    sample, preprocessing=preprocessing, spec=spec, raw_index=raw_idx
                )
            )
        except Exception as exc:
            tracker.record(exc, raw_idx, sample)

    return frames, tracker.summary(requested, len(frames))


def _class_row(summary: Mapping[int, ClassIoU], class_id: int) -> dict[str, Any]:
    v = summary[class_id]
    return {"iou": v.iou, "intersection": v.intersection, "union": v.union}


def _ablation_row_score(row: Mapping[str, Any]) -> tuple[float, float]:
    veh, road = row["classes"].get("10", {}), row["classes"].get("7", {})
    return (
        -1.0 if int(veh.get("union", 0)) == 0 else float(veh.get("iou", 0.0)),
        -1.0 if int(road.get("union", 0)) == 0 else float(road.get("iou", 0.0)),
    )


def _meta_float(metadata: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    try:
        val = (metadata or {}).get(key)
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _meta_int(metadata: Mapping[str, Any] | None, key: str, default: int = 0) -> int:
    try:
        val = (metadata or {}).get(key)
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _metadata_depth_coverage(
    metadata: Mapping[str, Any] | None, grid: SemanticOccupancyGrid
) -> float:
    if metadata is not None:
        depth_summary = metadata.get("depth_summary")
        if isinstance(depth_summary, Mapping):
            for k in ("coverage", "valid_fraction", "valid_ratio", "observed_fraction"):
                if k in depth_summary:
                    try:
                        return float(depth_summary[k])
                    except (TypeError, ValueError):
                        break
    return float(np.count_nonzero(grid.occupied) / grid.occupied.size)


def _yaw_delta_degrees(frames: Sequence[OccupancyFrame], anchor_index: int) -> float:
    if anchor_index <= 0:
        return 0.0
    curr_yaw = _meta_float(frames[anchor_index].metadata, "rotation_yaw", 0.0)
    prev_yaw = _meta_float(frames[anchor_index - 1].metadata, "rotation_yaw", curr_yaw)
    return abs(float((curr_yaw - prev_yaw + 180.0) % 360.0 - 180.0))


def _scenario_iou_payload(
    baseline_summary: Mapping[int, ClassIoU],
    temporal_summary: Mapping[int, ClassIoU],
    *,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    return {
        "classes": [
            {
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "baseline_iou": b.iou,
                "temporal_iou": t.iou,
                "delta": t.iou - b.iou,
                "baseline_intersection": b.intersection,
                "baseline_union": b.union,
                "temporal_intersection": t.intersection,
                "temporal_union": t.union,
                "informative": b.union > 0 or t.union > 0,
            }
            for cid in sorted(baseline_summary)
            for b, t in [(baseline_summary[cid], temporal_summary[cid])]
        ]
    }


def _evaluate_anchor_candidate(
    *,
    frames: Sequence[OccupancyFrame],
    anchor_idx: int,
    past_window: int,
    future_window: int,
    spec: OccupancyGridSpec,
    fusion_weights: Sequence[float],
    occupancy_threshold: float,
    class_ids: Sequence[int],
    sparse_threshold: float | None,
    config: ScenarioMemoryEvalConfig,
) -> _ScenarioMemoryCandidate | None:
    """Evaluate one anchor frame for scenario slice inclusion and IoU."""
    anchor = frames[anchor_idx]
    past_frames = frames[anchor_idx - past_window + 1 : anchor_idx + 1]
    future_frames = frames[anchor_idx : anchor_idx + future_window]

    target = union_grids_in_ego_frame(future_frames, target_ego_pose=anchor.ego_pose, spec=spec)
    fused = fuse_occupancy_frames(tuple(reversed(past_frames)), weights=fusion_weights)
    temporal = fused_to_semantic_grid(fused, occupancy_threshold=occupancy_threshold)

    b_veh = anchor.grid.occupied & (anchor.grid.semantic == VEHICLE_CLASS_ID)
    t_veh = target.occupied & (target.semantic == VEHICLE_CLASS_ID)
    scene_veh_union = int(np.count_nonzero(b_veh | t_veh))
    new_veh_voxels = int(np.count_nonzero(t_veh & ~b_veh))
    meta = anchor.metadata or {}
    coverage = _metadata_depth_coverage(meta, anchor.grid)
    abs_steer = abs(_meta_float(meta, "steer", 0.0))
    yaw_delta = _yaw_delta_degrees(frames, anchor_idx)
    nearby = _meta_int(meta, "nearby_vehicles_50m", 0)
    tot_veh = _meta_int(meta, "total_npc_vehicles", 0)
    disagreement = occupancy_disagreement_rate(
        anchor.grid, fused, occupancy_threshold=occupancy_threshold
    )

    slice_names: list[str] = []
    if scene_veh_union >= config.min_vehicle_union and new_veh_voxels >= config.min_vehicle_union:
        slice_names.append("vehicle_change")
    if sparse_threshold is not None and coverage <= sparse_threshold:
        slice_names.append("sparse_depth")
    if (
        abs_steer >= config.turn_steer_threshold
        or yaw_delta >= config.turn_yaw_delta_threshold_degrees
    ):
        slice_names.append("turning")
    if max(nearby, tot_veh) >= config.dense_traffic_min_nearby:
        slice_names.append("dense_traffic")
    if disagreement > config.disagreement_threshold:
        slice_names.append("high_disagreement")

    if not slice_names:
        return None

    return _ScenarioMemoryCandidate(
        anchor=ScenarioMemoryAnchor(
            anchor_index=anchor_idx,
            run_id=meta.get("run_id"),
            frame=meta.get("frame"),
            slice_names=tuple(slice_names),
            depth_coverage=float(coverage),
            abs_steer=float(abs_steer),
            yaw_delta_degrees=float(yaw_delta),
            nearby_vehicles_50m=nearby,
            total_npc_vehicles=tot_veh,
            disagreement_rate=float(disagreement),
            scene_vehicle_union=scene_veh_union,
            new_vehicle_voxel_count=new_veh_voxels,
        ),
        baseline=anchor.grid,
        temporal=temporal,
        target=target,
        baseline_iou=compute_semantic_iou(
            anchor.grid, target, class_ids=class_ids, occupancy_threshold=occupancy_threshold
        ),
        temporal_iou=compute_semantic_iou(
            fused, target, class_ids=class_ids, occupancy_threshold=occupancy_threshold
        ),
    )


def score_scenario_memory_candidates(
    frames: Sequence[OccupancyFrame],
    *,
    config: ScenarioMemoryEvalConfig | None = None,
) -> tuple[list[_ScenarioMemoryCandidate], dict[str, Any]]:
    """Score all context-valid anchors and assign heuristic scenario labels."""
    cfg = config or ScenarioMemoryEvalConfig()
    if not frames:
        return [], {"candidate_count": 0, "sparse_depth_threshold": None}

    spec = cfg.grid_spec or frames[0].grid.spec
    anchors = _all_context_valid_anchor_indices(
        frame_count=len(frames), past_window=cfg.past_window, future_window=cfg.future_window
    )
    if not anchors:
        return [], {"candidate_count": 0, "sparse_depth_threshold": None}

    coverages = [_metadata_depth_coverage(frames[i].metadata, frames[i].grid) for i in anchors]
    sparse_thresh = float(np.quantile(coverages, cfg.sparse_depth_quantile)) if coverages else None

    candidates: list[_ScenarioMemoryCandidate] = []
    for a_idx in anchors:
        candidate = _evaluate_anchor_candidate(
            frames=frames,
            anchor_idx=a_idx,
            past_window=cfg.past_window,
            future_window=cfg.future_window,
            spec=spec,
            fusion_weights=cfg.fusion_weights,
            occupancy_threshold=cfg.occupancy_threshold,
            class_ids=cfg.class_ids,
            sparse_threshold=sparse_thresh,
            config=cfg,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates, {
        "candidate_count": len(anchors),
        "labeled_candidate_count": len(candidates),
        "sparse_depth_threshold": sparse_thresh,
    }


def _select_candidates_by_slice(
    candidates: Sequence[_ScenarioMemoryCandidate],
    *,
    anchors_per_slice: int,
) -> dict[str, list[_ScenarioMemoryCandidate]]:
    selected: dict[str, list[_ScenarioMemoryCandidate]] = {
        s: [] for s in SCENARIO_MEMORY_SLICE_NAMES
    }
    for c in candidates:
        for s in c.anchor.slice_names:
            if len(selected[s]) < anchors_per_slice:
                selected[s].append(c)
    return selected


def _write_selected_anchors_jsonl(
    path: Path,
    selected_by_slice: Mapping[str, Sequence[_ScenarioMemoryCandidate]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for slice_name, candidates in selected_by_slice.items():
            for c in candidates:
                handle.write(json.dumps(c.anchor.to_json_record(slice_name=slice_name)) + "\n")
    return path


def _write_standard_artifacts(
    output_dir: Path,
    prefix: str,
    baseline_summary: Mapping[int, ClassIoU],
    temporal_summary: Mapping[int, ClassIoU],
    evaluator: ShadowModeEvaluator,
    class_names: Mapping[int, str],
) -> tuple[Path, Path, Path, Path]:
    """Write standard baseline/temporal IoU summaries and shadow mode records/clusters."""
    return (
        write_iou_summary_json(
            baseline_summary,
            output_dir / f"{prefix}_baseline_iou_summary.json",
            class_names=class_names,
        ),
        write_iou_summary_json(
            temporal_summary,
            output_dir / f"{prefix}_temporal_iou_summary.json",
            class_names=class_names,
        ),
        evaluator.write_records_jsonl(output_dir / f"{prefix}_shadow_records.jsonl"),
        evaluator.write_cluster_summary_json(output_dir / f"{prefix}_shadow_clusters.json"),
    )


def _base_eval_summary(
    eval_name: str,
    cfg: Any,
    class_names: Mapping[int, str],
    bev_images: Sequence[Path],
    vis_errors: Mapping[str, int],
    **extra: Any,
) -> dict[str, Any]:
    """Construct a shared evaluation summary payload."""
    return {
        "evaluation_name": eval_name,
        "dataset_name": cfg.dataset_name,
        "split": cfg.split,
        "occupancy_threshold": cfg.occupancy_threshold,
        "disagreement_threshold": cfg.disagreement_threshold,
        "class_ids": {str(cid): class_names.get(cid, str(cid)) for cid in cfg.class_ids},
        "visualization_errors": dict(vis_errors),
        "bev_images": [str(p) for p in bev_images],
        **extra,
    }


def run_scenario_memory_evaluation(
    *,
    config: ScenarioMemoryEvalConfig | None = None,
    raw_samples: Iterable[Mapping[str, Any]] | None = None,
    frames: Sequence[OccupancyFrame] | None = None,
) -> ScenarioMemoryEvalArtifacts:
    """Evaluate future-pseudo IoU on anchors where temporal memory should matter."""
    cfg = config or ScenarioMemoryEvalConfig()
    class_names = cfg.class_names or CARLA_CLASS_NAMES
    out_dir = Path(cfg.output_dir)
    bev_dir = out_dir / "scenario_memory_bev"
    col_summary: dict[str, Any] = {}

    if frames is None:
        frames, col_summary = _collect_occupancy_frames(
            config=cfg, raw_samples=raw_samples, max_raw_frames=cfg.max_scan_frames
        )
    else:
        col_summary = _ErrorTracker().summary(len(frames), len(frames))

    candidates, cand_summary = score_scenario_memory_candidates(frames, config=cfg)
    selected_by_slice = _select_candidates_by_slice(
        candidates, anchors_per_slice=cfg.anchors_per_slice
    )

    by_slice_payload: dict[str, Any] = {
        "evaluation_name": "scenario_memory_future_pseudo_iou",
        "target_proxy": (
            "future union of projected-depth semantic occupancy grids transformed into "
            "the anchor ego frame; not dense simulator 3D ground truth"
        ),
        "class_ids": {str(cid): class_names.get(cid, str(cid)) for cid in cfg.class_ids},
        "slices": {},
    }
    bev_images: list[Path] = []
    vis_errors: dict[str, int] = {}

    for s_name, selected in selected_by_slice.items():
        by_slice_payload["slices"][s_name] = {
            "selected_anchor_count": len(selected),
            **_scenario_iou_payload(
                summarize_iou([c.baseline_iou for c in selected]),
                summarize_iou([c.temporal_iou for c in selected]),
                class_names=class_names,
            ),
        }
        for c in selected[: cfg.bev_frame_count_per_slice]:
            lbl = c.anchor.frame or c.anchor.anchor_index
            _try_save_bev(
                baseline=c.baseline,
                temporal=c.temporal,
                target=c.target,
                output_path=bev_dir / s_name / f"{s_name}_{c.anchor.anchor_index:05d}_{lbl}.png",
                title=f"{s_name}: frame {lbl} future pseudo-label occupancy",
                class_names=class_names,
                panel_titles=("Baseline t", "Past temporal fusion", "Future pseudo-label"),
                error_tracker=vis_errors,
                images=bev_images,
            )

    selected_path = _write_selected_anchors_jsonl(
        out_dir / "scenario_memory_selected_anchors.jsonl", selected_by_slice
    )
    by_slice_path = _write_json(out_dir / "scenario_memory_by_slice_iou.json", by_slice_payload)
    summary_path = _write_json(
        out_dir / "scenario_memory_eval_summary.json",
        _base_eval_summary(
            "scenario_memory_future_pseudo_iou",
            cfg,
            class_names,
            bev_images,
            vis_errors,
            max_scan_frames=cfg.max_scan_frames,
            anchors_per_slice=cfg.anchors_per_slice,
            past_window=cfg.past_window,
            future_window=cfg.future_window,
            fusion_weights=list(cfg.fusion_weights),
            slice_names=list(SCENARIO_MEMORY_SLICE_NAMES),
            selected_anchor_counts={s: len(sel) for s, sel in selected_by_slice.items()},
            selection_thresholds={
                "min_scene_vehicle_union": cfg.min_vehicle_union,
                "sparse_depth_quantile": cfg.sparse_depth_quantile,
                "sparse_depth_threshold": cand_summary["sparse_depth_threshold"],
                "turn_steer_threshold": cfg.turn_steer_threshold,
                "turn_yaw_delta_threshold_degrees": cfg.turn_yaw_delta_threshold_degrees,
                "dense_traffic_min_nearby": cfg.dense_traffic_min_nearby,
            },
            **cand_summary,
            **col_summary,
        ),
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
    cfg = config or FuturePseudoEvalConfig()
    spec = cfg.grid_spec or OccupancyGridSpec()
    class_names = cfg.class_names or CARLA_CLASS_NAMES
    out_dir = Path(cfg.output_dir)
    bev_dir = out_dir / "future_pseudo_bev"
    align_dir = out_dir / "alignment"

    frames, col_summary = _collect_occupancy_frames(config=cfg, raw_samples=raw_samples)
    anchors = _anchor_indices(
        frame_count=len(frames),
        anchor_count=cfg.anchor_count,
        past_window=cfg.past_window,
        future_window=cfg.future_window,
    )
    if not anchors:
        _write_json(
            out_dir / "future_pseudo_eval_summary.json",
            {
                "evaluation_name": "future_pseudo_label_semantic_occupancy_iou",
                "anchor_count": 0,
                "skipped_anchor_count": max(cfg.anchor_count, 1),
                "reason": "insufficient processed context for requested past/future windows",
                **col_summary,
            },
        )
        raise RuntimeError("No anchors had enough past and future context for future-pseudo IoU")

    target_by_anchor: dict[int, SemanticOccupancyGrid] = {}
    baseline_iou_results: list[Mapping[int, ClassIoU]] = []
    ablation_iou_results: dict[tuple[str, float], list[Mapping[int, ClassIoU]]] = {
        (p.name, th): [] for p in cfg.fusion_profiles for th in cfg.thresholds
    }
    shadow_eval = ShadowModeEvaluator(
        disagreement_threshold=cfg.disagreement_threshold,
        occupancy_threshold=cfg.occupancy_threshold,
        class_ids=cfg.class_ids,
        class_names=class_names,
    )

    bev_images: list[Path] = []
    align_images: list[Path] = []
    align_records: list[dict[str, Any]] = []
    vis_errors: dict[str, int] = {}

    for pos, a_idx in enumerate(anchors):
        anchor = frames[a_idx]
        past_rev = tuple(reversed(frames[a_idx - cfg.past_window + 1 : a_idx + 1]))
        target = union_grids_in_ego_frame(
            frames[a_idx : a_idx + cfg.future_window], target_ego_pose=anchor.ego_pose, spec=spec
        )
        target_by_anchor[a_idx] = target
        baseline_iou_results.append(
            compute_semantic_iou(anchor.grid, target, class_ids=cfg.class_ids)
        )

        for prof in cfg.fusion_profiles:
            fused = fuse_occupancy_frames(past_rev, weights=prof.weights)
            for th in cfg.thresholds:
                ablation_iou_results[(prof.name, th)].append(
                    compute_semantic_iou(
                        fused, target, class_ids=cfg.class_ids, occupancy_threshold=th
                    )
                )

        default_fused: FusedOccupancyGrid | None = None
        for prof in cfg.fusion_profiles:
            fused = fuse_occupancy_frames(past_rev, weights=prof.weights)
            if default_fused is None:
                default_fused = fused
            for th in cfg.thresholds:
                ablation_iou_results[(prof.name, th)].append(
                    compute_semantic_iou(
                        fused, target, class_ids=cfg.class_ids, occupancy_threshold=th
                    )
                )

        assert default_fused is not None
        shadow_eval.evaluate_frame(
            baseline=anchor.grid, improved=default_fused, metadata=anchor.metadata or {}
        )

        if len(bev_images) < cfg.bev_frame_count:
            temporal = fused_to_semantic_grid(
                default_fused, occupancy_threshold=cfg.occupancy_threshold
            )
            lbl = (anchor.metadata or {}).get("frame", a_idx)
            _try_save_bev(
                baseline=anchor.grid,
                temporal=temporal,
                target=target,
                output_path=bev_dir / f"future_pseudo_bev_{pos:05d}_{lbl}.png",
                title=f"Frame {lbl} future pseudo-label occupancy",
                class_names=class_names,
                panel_titles=("Baseline t", "Past temporal fusion", "Future pseudo-label"),
                error_tracker=vis_errors,
                images=bev_images,
            )

        if len(align_images) < cfg.alignment_frame_count and a_idx > 0:
            prev_aligned = union_grids_in_ego_frame(
                (frames[a_idx - 1],), target_ego_pose=anchor.ego_pose, spec=spec
            )
            lbl = (anchor.metadata or {}).get("frame", a_idx)
            img = _try_save_bev(
                baseline=anchor.grid,
                temporal=prev_aligned,
                target=target,
                output_path=align_dir / f"alignment_frame_{pos:05d}_{lbl}.png",
                title=f"Frame {lbl} temporal alignment diagnostic",
                class_names=class_names,
                panel_titles=("Current t", "Previous aligned to t", "Future target in t"),
                error_tracker=vis_errors,
                images=align_images,
            )
            align_records.append(
                {
                    "anchor_position": pos,
                    "anchor_index": a_idx,
                    "frame": lbl,
                    "previous_overlap_ratio": _occupied_overlap_ratio(anchor.grid, prev_aligned),
                    "future_overlap_ratio": _occupied_overlap_ratio(anchor.grid, target),
                    "previous_near_overlap_ratio": _occupied_near_overlap_ratio(
                        anchor.grid, prev_aligned, tolerance_voxels=cfg.alignment_tolerance_voxels
                    ),
                    "future_near_overlap_ratio": _occupied_near_overlap_ratio(
                        anchor.grid, target, tolerance_voxels=cfg.alignment_tolerance_voxels
                    ),
                    **({"image_path": str(img)} if img else {}),
                }
            )

    ablation_rows = [
        {
            "profile": p_name,
            "occupancy_threshold": th,
            "classes": {
                str(cid): {
                    "class_name": class_names.get(cid, str(cid)),
                    **_class_row(summarize_iou(res), cid),
                }
                for cid in cfg.class_ids
            },
        }
        for (p_name, th), res in ablation_iou_results.items()
    ]
    best_row = max(ablation_rows, key=_ablation_row_score)
    best_prof = next(p for p in cfg.fusion_profiles if p.name == best_row["profile"])
    best_th = float(cast(float, best_row["occupancy_threshold"]))

    temporal_summary = summarize_iou(
        [
            compute_semantic_iou(
                fuse_occupancy_frames(
                    tuple(reversed(frames[i - cfg.past_window + 1 : i + 1])),
                    weights=best_prof.weights,
                ),
                target_by_anchor[i],
                class_ids=cfg.class_ids,
                occupancy_threshold=best_th,
            )
            for i in anchors
        ]
    )

    base_p, temp_p, rec_p, clus_p = _write_standard_artifacts(
        out_dir,
        "future_pseudo",
        summarize_iou(baseline_iou_results),
        temporal_summary,
        shadow_eval,
        class_names,
    )
    ablation_p = _write_json(
        out_dir / "fusion_ablation_summary.json",
        {
            "selection_rule": "primary vehicle IoU, secondary road IoU; empty-union classes score -1",
            "selected_profile": best_prof.name,
            "selected_weights": list(best_prof.weights),
            "selected_occupancy_threshold": best_th,
            "rows": ablation_rows,
        },
    )
    align_stats = {
        f"mean_{k}": float(np.mean([r[k] for r in align_records])) if align_records else None
        for k in (
            "previous_overlap_ratio",
            "future_overlap_ratio",
            "previous_near_overlap_ratio",
            "future_near_overlap_ratio",
        )
    }
    align_p = _write_json(
        out_dir / "alignment_summary.json",
        {
            "diagnostic": "adjacent and future occupancy transformed into anchor ego frame",
            "frame_count": len(align_records),
            "near_overlap_tolerance_voxels": cfg.alignment_tolerance_voxels,
            **align_stats,
            "records": align_records,
            "image_paths": [str(p) for p in align_images],
        },
    )
    summary_p = _write_json(
        out_dir / "future_pseudo_eval_summary.json",
        _base_eval_summary(
            "future_pseudo_label_semantic_occupancy_iou",
            cfg,
            class_names,
            bev_images,
            vis_errors,
            requested_anchor_count=cfg.anchor_count,
            processed_anchor_count=len(anchors),
            skipped_anchor_count=max(cfg.anchor_count - len(anchors), 0),
            past_window=cfg.past_window,
            future_window=cfg.future_window,
            selected_profile=best_prof.name,
            selected_weights=list(best_prof.weights),
            selected_occupancy_threshold=best_th,
            target_proxy=(
                "future union of projected-depth semantic occupancy grids from frames "
                "t through t+future_window-1 transformed into frame t; this is not dense "
                "simulator 3D ground truth"
            ),
            alignment_images=[str(p) for p in align_images],
            **col_summary,
        ),
    )

    return FuturePseudoEvalArtifacts(
        baseline_iou=base_p,
        temporal_iou=temp_p,
        eval_summary=summary_p,
        shadow_records=rec_p,
        shadow_clusters=clus_p,
        alignment_summary=align_p,
        ablation_summary=ablation_p,
        bev_images=tuple(bev_images),
        alignment_images=tuple(align_images),
    )


def run_bounded_carla_iou_evaluation(
    *,
    config: Phase2CarlaEvalConfig | None = None,
    raw_samples: Iterable[Mapping[str, Any]] | None = None,
) -> Phase2CarlaEvalArtifacts:
    """Run bounded Phase 2 evaluation and write notebook-readable artifacts."""
    cfg = config or Phase2CarlaEvalConfig()
    preprocessing = cfg.preprocessing or build_occupancy_preprocessing_config()
    spec = cfg.grid_spec or OccupancyGridSpec()
    class_names = cfg.class_names or CARLA_CLASS_NAMES
    out_dir = Path(cfg.output_dir)
    bev_dir = out_dir / "bev"
    samples = (
        raw_samples
        if raw_samples is not None
        else stream_huggingface_samples(cfg.dataset_name, cfg.split)
    )

    buffer = TemporalFusionBuffer()
    evaluator = ShadowModeEvaluator(
        disagreement_threshold=cfg.disagreement_threshold,
        occupancy_threshold=cfg.occupancy_threshold,
        class_ids=cfg.class_ids,
        class_names=class_names,
    )
    baseline_results: list[Mapping[int, ClassIoU]] = []
    temporal_results: list[Mapping[int, ClassIoU]] = []
    bev_images: list[Path] = []
    vis_errors: dict[str, int] = {}
    tracker = _ErrorTracker()
    processed_count = 0

    for idx, sample in enumerate(islice(samples, cfg.max_frames)):
        try:
            frame = _frame_from_sample(sample, preprocessing=preprocessing, spec=spec)
            fused = buffer.add(frame)
            temporal = fused_to_semantic_grid(fused, occupancy_threshold=cfg.occupancy_threshold)

            baseline_results.append(
                compute_semantic_iou(frame.grid, frame.grid, class_ids=cfg.class_ids)
            )
            temporal_results.append(
                compute_semantic_iou(
                    fused,
                    frame.grid,
                    class_ids=cfg.class_ids,
                    occupancy_threshold=cfg.occupancy_threshold,
                )
            )
            evaluator.evaluate_frame(baseline=frame.grid, improved=fused, metadata=frame.metadata)
            processed_count += 1

            if len(bev_images) < cfg.bev_frame_count or (
                evaluator.records[-1].flagged and len(bev_images) < cfg.bev_frame_count + 2
            ):
                lbl = (frame.metadata or {}).get("frame", idx)
                _try_save_bev(
                    baseline=frame.grid,
                    temporal=temporal,
                    target=frame.grid,
                    output_path=bev_dir / f"bev_frame_{idx:05d}_{lbl}.png",
                    title=f"Frame {lbl} projected-depth proxy occupancy",
                    class_names=class_names,
                    error_tracker=vis_errors,
                    images=bev_images,
                )
        except Exception as exc:
            tracker.record(exc, idx, sample)

    tracker.raise_if_empty(processed_count)

    base_p, temp_p, rec_p, clus_p = _write_standard_artifacts(
        out_dir,
        "carla",
        summarize_iou(baseline_results),
        summarize_iou(temporal_results),
        evaluator,
        class_names,
    )
    summary_p = _write_json(
        out_dir / "carla_evaluation_run_summary.json",
        _base_eval_summary(
            "current_frame_proxy_agreement_diagnostic",
            cfg,
            class_names,
            bev_images,
            vis_errors,
            requested_max_frames=cfg.max_frames,
            processed_frame_count=processed_count,
            target_proxy=(
                "current-frame occupancy grid from LiDAR-converted projected depth and "
                "front semantic segmentation; this is a diagnostic agreement check, not "
                "final IoU or dense simulator 3D ground truth"
            ),
            free_space_note=(
                "Free space IoU is not reported because this occupancy representation stores "
                "observed occupied voxels and does not raycast explicit free-space labels."
            ),
            **tracker.summary(cfg.max_frames, processed_count),
        ),
    )

    return Phase2CarlaEvalArtifacts(base_p, temp_p, rec_p, clus_p, summary_p, tuple(bev_images))


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
