"""Offline hero-figure renderer for the README.

Builds small synthetic occupancy grids that mimic CARLA validation anchor 475
in the high-disagreement slice, then renders them with the fixed
``save_bev_comparison`` so the README figure stays readable without
downloading the HuggingFace dataset.

Provenance: synthetic stand-in, not measured voxels. Panel shapes and the
baseline hole pattern were chosen to resemble the real 1,003 vs 1,163
vehicle-voxel recovery reported in ``docs/phase2_closeout.md``. Real eval
figures written to ``outputs/`` (gitignored) use the same plotting code, so a
future dataset run reproduces this style automatically.

Run: ``uv run python scripts/make_readme_bev.py``
"""

from pathlib import Path

import numpy as np

from src.evaluation.phase2_carla_eval import CARLA_CLASS_NAMES, save_bev_comparison
from src.perception.occupancy_grid import UNKNOWN_CLASS, OccupancyGridSpec, SemanticOccupancyGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "bev_occlusion_recovery.png"

ROAD_ID = 7
VEHICLE_ID = 10
ROAD_LINE_ID = 6
SIDEWALK_ID = 8

SPEC = OccupancyGridSpec()


def _empty_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = SPEC.shape
    occupied = np.zeros(shape, dtype=bool)
    semantic = np.full(shape, UNKNOWN_CLASS, dtype=np.int16)
    counts = np.zeros(shape, dtype=np.uint16)
    return occupied, semantic, counts


def _paint_box(
    occupied: np.ndarray,
    semantic: np.ndarray,
    counts: np.ndarray,
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    z_range: tuple[int, int],
    label: int,
    skip_mask: np.ndarray | None = None,
) -> None:
    for xi in range(x_range[0], x_range[1]):
        for yi in range(y_range[0], y_range[1]):
            for zi in range(z_range[0], z_range[1]):
                if skip_mask is not None and skip_mask[xi, yi]:
                    continue
                occupied[xi, yi, zi] = True
                semantic[xi, yi, zi] = label
                counts[xi, yi, zi] = 1


def _road_and_context(occupied: np.ndarray, semantic: np.ndarray, counts: np.ndarray) -> None:
    # Horizontal road band with lane markings and sidewalks for orientation.
    _paint_box(
        occupied,
        semantic,
        counts,
        x_range=(55, 150),
        y_range=(88, 112),
        z_range=(0, 2),
        label=ROAD_ID,
    )
    _paint_box(
        occupied,
        semantic,
        counts,
        x_range=(55, 150),
        y_range=(99, 101),
        z_range=(2, 3),
        label=ROAD_LINE_ID,
    )
    _paint_box(
        occupied,
        semantic,
        counts,
        x_range=(55, 150),
        y_range=(84, 88),
        z_range=(0, 2),
        label=SIDEWALK_ID,
    )
    _paint_box(
        occupied,
        semantic,
        counts,
        x_range=(55, 150),
        y_range=(112, 116),
        z_range=(0, 2),
        label=SIDEWALK_ID,
    )


def _vehicle_holes(kind: str) -> np.ndarray:
    holes = np.zeros((SPEC.shape[0], SPEC.shape[1]), dtype=bool)
    if kind == "baseline":
        # One clean occlusion stripe plus two block dropouts: reads as
        # occlusion at README size instead of random static.
        holes[123:126, 96:112] = True
        holes[118:122, 100:104] = True
        holes[128:131, 106:110] = True
    elif kind == "temporal":
        # Two small residual holes; everything else recovered.
        holes[130:131, 109:110] = True
        holes[119:120, 97:98] = True
    return holes


def _build_grid(kind: str) -> SemanticOccupancyGrid:
    occupied, semantic, counts = _empty_grids()
    _road_and_context(occupied, semantic, counts)
    _paint_box(
        occupied,
        semantic,
        counts,
        x_range=(118, 136),
        y_range=(96, 110),
        z_range=(2, 7),
        label=VEHICLE_ID,
        skip_mask=_vehicle_holes(kind) if kind in {"baseline", "temporal"} else None,
    )
    return SemanticOccupancyGrid(SPEC, occupied, semantic, counts)


def main() -> Path:
    baseline = _build_grid("baseline")
    temporal = _build_grid("temporal")
    target = _build_grid("target")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_bev_comparison(
        baseline=baseline,
        temporal=temporal,
        target=target,
        output_path=OUTPUT_PATH,
        title="Occlusion recovery, anchor 475",
        subtitle="High-disagreement slice: single frame vs 5-frame memory vs denser future target",
        class_names=CARLA_CLASS_NAMES,
        panel_titles=("Single frame", "5-frame memory", "Denser future target"),
        highlight_recovered=True,
    )
    baseline_n = int(np.sum((baseline.semantic == VEHICLE_ID) & baseline.occupied))
    temporal_n = int(np.sum((temporal.semantic == VEHICLE_ID) & temporal.occupied))
    target_n = int(np.sum((target.semantic == VEHICLE_ID) & target.occupied))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"vehicle voxels baseline/temporal/target: {baseline_n}/{temporal_n}/{target_n}")
    print("Synthetic provenance: mimics frame 475 hole pattern, not measured data.")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
