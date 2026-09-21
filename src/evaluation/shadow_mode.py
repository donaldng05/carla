"""Shadow-mode comparison between baseline and temporally fused occupancy grids."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.iou import compute_semantic_iou, iou_table
from src.perception.occupancy_grid import SemanticOccupancyGrid
from src.perception.temporal_fusion import FusedOccupancyGrid


@dataclass(frozen=True)
class ShadowFrameResult:
    """One shadow-mode comparison record."""

    run_id: str | None
    frame: int | None
    disagreement_rate: float
    flagged: bool
    metadata: dict[str, Any]
    per_class_iou: list[dict[str, float | int | str]] = field(default_factory=list)

    def to_json_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "frame": self.frame,
            "disagreement_rate": self.disagreement_rate,
            "flagged": self.flagged,
            "metadata": self.metadata,
            "per_class_iou": self.per_class_iou,
        }


def occupancy_disagreement_rate(
    baseline: SemanticOccupancyGrid,
    improved: FusedOccupancyGrid,
    *,
    occupancy_threshold: float = 0.5,
) -> float:
    """Return disagreement over the observed occupied union.

    Sparse LiDAR occupancy grids are mostly empty. Normalizing by the full grid
    hides meaningful differences, so shadow-mode diagnostics use voxels that
    either model marks occupied as the denominator.
    """

    if baseline.occupied.shape != improved.occupancy_score.shape:
        raise ValueError("baseline and improved occupancy grids must have matching shapes")
    improved_occupied = improved.occupancy_score >= occupancy_threshold
    observed_union = baseline.occupied | improved_occupied
    union_count = int(np.count_nonzero(observed_union))
    if union_count == 0:
        return 0.0
    disagreement_count = int(
        np.count_nonzero((baseline.occupied != improved_occupied) & observed_union)
    )
    return float(disagreement_count / union_count)


def extract_shadow_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the scenario fields used by Phase 2 failure-mode clustering."""

    if metadata is None:
        return {}

    keys = (
        "index",
        "timestamp",
        "speed_kmh",
        "map_name",
        "weather_cloudiness",
        "weather_precipitation",
        "weather_fog_density",
        "weather_sun_altitude_angle",
        "nearby_vehicles_50m",
        "total_npc_vehicles",
        "total_npc_walkers",
        "depth_summary",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


class ShadowModeEvaluator:
    """Evaluate baseline single-frame occupancy against temporal fusion in shadow mode."""

    def __init__(
        self,
        *,
        disagreement_threshold: float = 0.05,
        occupancy_threshold: float = 0.5,
        class_ids: Iterable[int] = (),
        class_names: Mapping[int, str] | None = None,
    ) -> None:
        self.disagreement_threshold = float(disagreement_threshold)
        self.occupancy_threshold = float(occupancy_threshold)
        self.class_ids = tuple(int(class_id) for class_id in class_ids)
        self.class_names = dict(class_names or {})
        self.records: list[ShadowFrameResult] = []

    def evaluate_frame(
        self,
        *,
        baseline: SemanticOccupancyGrid,
        improved: FusedOccupancyGrid,
        metadata: Mapping[str, Any] | None = None,
    ) -> ShadowFrameResult:
        disagreement = occupancy_disagreement_rate(
            baseline,
            improved,
            occupancy_threshold=self.occupancy_threshold,
        )
        metadata_dict = dict(metadata or {})
        per_class_iou: list[dict[str, float | int | str]] = []
        if self.class_ids:
            per_class_iou = iou_table(
                compute_semantic_iou(
                    improved,
                    baseline,
                    class_ids=self.class_ids,
                    occupancy_threshold=self.occupancy_threshold,
                ),
                class_names=self.class_names,
            )

        result = ShadowFrameResult(
            run_id=None if metadata is None else metadata.get("run_id"),
            frame=None if metadata is None else metadata.get("frame"),
            disagreement_rate=disagreement,
            flagged=disagreement > self.disagreement_threshold,
            metadata=extract_shadow_metadata(metadata_dict),
            per_class_iou=per_class_iou,
        )
        self.records.append(result)
        return result

    def flagged_records(self) -> list[ShadowFrameResult]:
        return [record for record in self.records if record.flagged]

    def cluster_flagged_by_condition(self) -> dict[str, dict[str, int]]:
        """Group flagged records into coarse weather, lighting, traffic, and speed buckets."""

        clusters: dict[str, dict[str, int]] = {
            "weather": {},
            "lighting": {},
            "traffic": {},
            "speed": {},
        }
        for record in self.flagged_records():
            metadata = record.metadata
            precipitation = float(metadata.get("weather_precipitation") or 0.0)
            fog = float(metadata.get("weather_fog_density") or 0.0)
            sun = float(metadata.get("weather_sun_altitude_angle") or 0.0)
            nearby = int(metadata.get("nearby_vehicles_50m") or 0)
            speed = float(metadata.get("speed_kmh") or 0.0)

            weather = "adverse" if precipitation > 10.0 or fog > 10.0 else "clear"
            lighting = "low_sun" if sun < 10.0 else "daylight"
            traffic = "dense" if nearby >= 10 else "light"
            speed_bucket = "high_speed" if speed >= 40.0 else "low_speed"

            for family, bucket in (
                ("weather", weather),
                ("lighting", lighting),
                ("traffic", traffic),
                ("speed", speed_bucket),
            ):
                clusters[family][bucket] = clusters[family].get(bucket, 0) + 1

        return clusters

    def write_records_jsonl(self, output_path: str | Path) -> Path:
        """Write one shadow-mode JSON record per evaluated frame."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_json_record()) + "\n")
        return path

    def write_cluster_summary_json(self, output_path: str | Path) -> Path:
        """Write a compact summary of high-disagreement condition buckets."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "flagged_frame_count": len(self.flagged_records()),
            "total_frame_count": len(self.records),
            "disagreement_threshold": self.disagreement_threshold,
            "clusters": self.cluster_flagged_by_condition(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
