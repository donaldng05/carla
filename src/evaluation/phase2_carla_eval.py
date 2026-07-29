"""Bounded streamed CARLA Phase 2 occupancy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from src.data.carla_dataset import (
    DEFAULT_DATASET_NAME,
    CarlaDataPreprocessingConfig,
    CarlaMultimodalDataset,
    build_occupancy_preprocessing_config,
)
from src.evaluation.iou import ClassIoU, compute_semantic_iou, summarize_iou, write_iou_summary_json
from src.evaluation.shadow_mode import ShadowModeEvaluator
from src.perception.modality_verification import estimate_front_camera_intrinsics
from src.perception.occupancy_grid import (
    UNKNOWN_CLASS,
    OccupancyGridSpec,
    SemanticOccupancyGrid,
    build_semantic_occupancy_grid,
)
from src.perception.temporal_fusion import FusedOccupancyGrid, OccupancyFrame, TemporalFusionBuffer
from src.transforms import carla_pose_to_matrix

CARLA_CLASS_NAMES = {
    7: "road",
    10: "vehicle",
    4: "pedestrian",
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
    panels = [
        ("Baseline", panel_arrays[0]),
        ("Temporal fusion", panel_arrays[1]),
        ("Target proxy", panel_arrays[2]),
    ]

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
            "evaluation_name": "bounded_projected_depth_semantic_occupancy_iou",
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
                "front semantic segmentation; this is not dense simulator 3D ground truth"
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
