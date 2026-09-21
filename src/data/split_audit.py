"""Split audit helpers for CARLA Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "immanuelpeter/carla-autopilot-multimodal-dataset"


def _load_hf_split(
    dataset_name: str,
    split: str,
    *,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
) -> Any:
    from datasets import load_dataset

    return load_dataset(
        dataset_name,
        split=split,
        cache_dir=None if cache_dir is None else str(cache_dir),
        streaming=streaming,
    )


def audit_split_run_ids(
    split_samples: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    run_key: str = "run_id",
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    """Audit whether any run IDs appear in more than one split."""

    runs_by_split: dict[str, set[str]] = {}
    inspected_counts: dict[str, int] = {}

    for split_name, samples in split_samples.items():
        run_ids: set[str] = set()
        inspected = 0
        for sample in samples:
            inspected += 1
            run_ids.add(str(sample.get(run_key, "unknown")))
            if max_samples_per_split is not None and inspected >= max_samples_per_split:
                break
        runs_by_split[split_name] = run_ids
        inspected_counts[split_name] = inspected

    seen_in_splits: dict[str, list[str]] = defaultdict(list)
    for split_name, run_ids in runs_by_split.items():
        for run_id in run_ids:
            seen_in_splits[run_id].append(split_name)

    overlaps = {
        run_id: sorted(split_names)
        for run_id, split_names in seen_in_splits.items()
        if len(split_names) > 1
    }

    return {
        "has_overlap": bool(overlaps),
        "overlaps": overlaps,
        "runs_by_split": {split: sorted(run_ids) for split, run_ids in runs_by_split.items()},
        "run_counts": {split: len(run_ids) for split, run_ids in runs_by_split.items()},
        "inspected_counts": inspected_counts,
    }


def audit_huggingface_split_run_ids(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    splits: Sequence[str] = ("train", "validation", "test"),
    cache_dir: str | Path | None = None,
    streaming: bool = True,
    max_samples_per_split: int | None = None,
    strategy: str = "datasets",
    **kwargs: Any,
) -> dict[str, Any]:
    """Audit HF split run boundaries using streaming datasets."""

    split_samples = {
        split: _load_hf_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        for split in splits
    }
    report = audit_split_run_ids(split_samples, max_samples_per_split=max_samples_per_split)
    report["audit_strategy"] = strategy
    report["is_definitive"] = max_samples_per_split is None
    report["notes"] = (
        "This audit used the Hugging Face datasets library directly. "
        "It is definitive only when the full split was inspected."
    )
    return report


def format_split_audit_report(
    report: Mapping[str, Any],
    *,
    dataset_name: str,
    splits: Sequence[str],
    streaming: bool,
    max_samples_per_split: int | None,
) -> str:
    """Format a human-readable split audit summary."""

    lines = [
        f"Dataset: {dataset_name}",
        f"Splits audited: {', '.join(splits)}",
        f"Audit strategy: {report.get('audit_strategy', 'datasets')}",
        f"Definitive audit: {bool(report.get('is_definitive', False))}",
        f"Streaming mode: {streaming}",
        (
            "Max samples per split: full split"
            if max_samples_per_split is None
            else f"Max samples per split: {max_samples_per_split}"
        ),
        f"Run overlap detected: {bool(report['has_overlap'])}",
    ]

    if "window_size" in report and "window_count" in report:
        lines.append(
            f"Sampling window: count={report['window_count']}, rows_per_window={report['window_size']}"
        )
    if report.get("notes"):
        lines.append(f"Notes: {report['notes']}")

    lines.extend(
        [
            "",
            "Per-split summary:",
        ]
    )

    for split in splits:
        inspected = report["inspected_counts"].get(split, 0)
        run_count = report["run_counts"].get(split, 0)
        run_ids = report["runs_by_split"].get(split, [])
        preview = ", ".join(run_ids[:10]) if run_ids else "(none found)"
        if len(run_ids) > 10:
            preview += ", ..."
        details = (
            f"- {split}: inspected={inspected}, unique_runs={run_count}, run_preview=[{preview}]"
        )
        if "split_sizes" in report:
            details += f", estimated_rows={report['split_sizes'].get(split, 0)}"
        if "observed_offsets" in report:
            details += f", offsets={report['observed_offsets'].get(split, [])}"
        lines.append(details)

    overlaps = report["overlaps"]
    if overlaps:
        lines.extend(["", "Overlapping run IDs:"])
        for run_id in sorted(overlaps):
            lines.append(f"- {run_id}: {', '.join(overlaps[run_id])}")
    else:
        lines.extend(["", "No overlapping run IDs were detected in the audited window."])

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit run_id overlap across Hugging Face dataset splits.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to audit.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Split names to audit.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face streaming mode for the exact datasets-based audit.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap per split for bounded audits.",
    )
    parser.add_argument(
        "--strategy",
        default="datasets",
        help="Split audit backend (defaults to 'datasets').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw report as JSON instead of the human-readable summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = audit_huggingface_split_run_ids(
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        max_samples_per_split=args.max_samples_per_split,
        strategy=args.strategy,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            format_split_audit_report(
                report,
                dataset_name=args.dataset_name,
                splits=tuple(args.splits),
                streaming=args.streaming,
                max_samples_per_split=args.max_samples_per_split,
            )
        )

    return 1 if report["has_overlap"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
