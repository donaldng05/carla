"""Generate deterministic Phase 2 smoke-evaluation artifacts.

This module validates that the IoU and shadow-mode reporting path produces
readable files before running the expensive streamed dataset evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.iou import compute_semantic_iou, summarize_iou, write_iou_summary_json
from src.evaluation.shadow_mode import ShadowModeEvaluator
from src.perception.occupancy_grid import OccupancyGridSpec, SemanticOccupancyGrid
from src.perception.temporal_fusion import OccupancyFrame, TemporalFusionBuffer

CLASS_NAMES = {
    1: "road",
    2: "vehicle",
    3: "pedestrian",
}
CLASS_IDS = tuple(CLASS_NAMES)


def _make_grid(
    spec: OccupancyGridSpec, voxels: list[tuple[int, int, int, int]]
) -> SemanticOccupancyGrid:
    occupied = np.zeros(spec.shape, dtype=bool)
    semantic = np.full(spec.shape, -1, dtype=np.int16)
    counts = np.zeros(spec.shape, dtype=np.uint16)
    for x_index, y_index, z_index, label in voxels:
        occupied[x_index, y_index, z_index] = True
        semantic[x_index, y_index, z_index] = label
        counts[x_index, y_index, z_index] = 1
    return SemanticOccupancyGrid(spec, occupied, semantic, counts)


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def run_phase2_smoke_eval(output_dir: str | Path = "outputs/phase2") -> dict[str, Path]:
    """Run a tiny deterministic Phase 2 evaluation and write inspectable artifacts."""

    output_path = Path(output_dir)
    spec = OccupancyGridSpec(
        voxel_size_m=1.0,
        x_range_m=(-3.0, 3.0),
        y_range_m=(-3.0, 3.0),
        z_range_m=(0.0, 2.0),
    )

    targets = [
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1), (4, 3, 1, 2)]),
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1), (4, 3, 1, 2)]),
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1), (4, 3, 1, 2), (1, 2, 0, 3)]),
    ]
    baselines = [
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1), (4, 3, 1, 2)]),
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1)]),
        _make_grid(spec, [(2, 3, 0, 1), (3, 3, 0, 1), (1, 2, 0, 3)]),
    ]
    metadata = [
        {
            "run_id": "phase2_smoke",
            "frame": 0,
            "speed_kmh": 20.0,
            "weather_precipitation": 0.0,
            "weather_fog_density": 0.0,
            "weather_sun_altitude_angle": 45.0,
            "nearby_vehicles_50m": 3,
        },
        {
            "run_id": "phase2_smoke",
            "frame": 1,
            "speed_kmh": 35.0,
            "weather_precipitation": 0.0,
            "weather_fog_density": 0.0,
            "weather_sun_altitude_angle": 35.0,
            "nearby_vehicles_50m": 5,
        },
        {
            "run_id": "phase2_smoke",
            "frame": 2,
            "speed_kmh": 48.0,
            "weather_precipitation": 25.0,
            "weather_fog_density": 0.0,
            "weather_sun_altitude_angle": 5.0,
            "nearby_vehicles_50m": 12,
        },
    ]

    buffer = TemporalFusionBuffer()
    evaluator = ShadowModeEvaluator(
        disagreement_threshold=0.01,
        occupancy_threshold=0.25,
        class_ids=CLASS_IDS,
        class_names=CLASS_NAMES,
    )
    baseline_iou = []
    temporal_iou = []

    for frame_index, baseline in enumerate(baselines):
        fused = buffer.add(
            OccupancyFrame(
                baseline,
                _pose(),
                metadata=metadata[frame_index],
            )
        )
        evaluator.evaluate_frame(
            baseline=baseline,
            improved=fused,
            metadata=metadata[frame_index],
        )
        baseline_iou.append(
            compute_semantic_iou(baseline, targets[frame_index], class_ids=CLASS_IDS)
        )
        temporal_iou.append(
            compute_semantic_iou(
                fused,
                targets[frame_index],
                class_ids=CLASS_IDS,
                occupancy_threshold=0.25,
            )
        )

    baseline_summary = summarize_iou(baseline_iou)
    temporal_summary = summarize_iou(temporal_iou)

    paths = {
        "baseline_iou": write_iou_summary_json(
            baseline_summary,
            output_path / "baseline_iou_summary.json",
            class_names=CLASS_NAMES,
        ),
        "temporal_iou": write_iou_summary_json(
            temporal_summary,
            output_path / "temporal_iou_summary.json",
            class_names=CLASS_NAMES,
        ),
        "shadow_records": evaluator.write_records_jsonl(output_path / "shadow_mode_records.jsonl"),
        "shadow_clusters": evaluator.write_cluster_summary_json(
            output_path / "shadow_mode_clusters.json"
        ),
    }
    return paths


def main() -> None:
    paths = run_phase2_smoke_eval()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
