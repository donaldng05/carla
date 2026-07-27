import json
from pathlib import Path

import numpy as np

from src.evaluation.shadow_mode import ShadowModeEvaluator, occupancy_disagreement_rate
from src.perception.occupancy_grid import OccupancyGridSpec, SemanticOccupancyGrid
from src.perception.temporal_fusion import FusedOccupancyGrid


def grids() -> tuple[SemanticOccupancyGrid, FusedOccupancyGrid]:
    spec = OccupancyGridSpec(voxel_size_m=1.0, x_range_m=(0, 2), y_range_m=(0, 2), z_range_m=(0, 1))
    baseline_occupied = np.array([[[True], [False]], [[False], [True]]])
    baseline_semantic = np.array([[[1], [-1]], [[-1], [2]]], dtype=np.int16)
    baseline = SemanticOccupancyGrid(
        spec,
        baseline_occupied,
        baseline_semantic,
        baseline_occupied.astype(np.uint16),
    )
    improved_score = np.array([[[1.0], [0.6]], [[0.0], [1.0]]])
    improved_semantic = np.array([[[1], [1]], [[-1], [2]]], dtype=np.int16)
    improved = FusedOccupancyGrid(spec, improved_score, improved_semantic, (1.0,))
    return baseline, improved


def test_occupancy_disagreement_rate_counts_binary_differences() -> None:
    baseline, improved = grids()

    disagreement = occupancy_disagreement_rate(baseline, improved)

    assert disagreement == 0.25


def test_shadow_mode_evaluator_flags_and_keeps_metadata() -> None:
    baseline, improved = grids()
    evaluator = ShadowModeEvaluator(disagreement_threshold=0.1, class_ids=(1, 2))

    result = evaluator.evaluate_frame(
        baseline=baseline,
        improved=improved,
        metadata={
            "run_id": "run_a",
            "frame": 42,
            "speed_kmh": 45.0,
            "weather_precipitation": 20.0,
            "weather_fog_density": 0.0,
            "weather_sun_altitude_angle": 5.0,
            "nearby_vehicles_50m": 12,
        },
    )

    assert result.flagged
    assert result.run_id == "run_a"
    assert result.frame == 42
    assert result.per_class_iou
    clusters = evaluator.cluster_flagged_by_condition()
    assert clusters["weather"]["adverse"] == 1
    assert clusters["lighting"]["low_sun"] == 1
    assert clusters["traffic"]["dense"] == 1
    assert clusters["speed"]["high_speed"] == 1


def test_shadow_mode_evaluator_writes_records_and_cluster_summary(tmp_path: Path) -> None:
    baseline, improved = grids()
    evaluator = ShadowModeEvaluator(disagreement_threshold=0.1)
    evaluator.evaluate_frame(baseline=baseline, improved=improved, metadata={"run_id": "run_a"})

    records_path = evaluator.write_records_jsonl(tmp_path / "shadow_mode_records.jsonl")
    summary_path = evaluator.write_cluster_summary_json(tmp_path / "shadow_mode_clusters.json")

    first_record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first_record["run_id"] == "run_a"
    assert summary["flagged_frame_count"] == 1
