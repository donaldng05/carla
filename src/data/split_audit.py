"""Split audit helpers for CARLA Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

DEFAULT_DATASET_NAME = "immanuelpeter/carla-autopilot-multimodal-dataset"
DEFAULT_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
DEFAULT_WINDOW_SIZE = 25
DEFAULT_WINDOW_COUNT = 5


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


def _build_datasets_server_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _fetch_datasets_server_json(
    path: str,
    *,
    params: Mapping[str, Any],
    datasets_server_url: str,
) -> Any:
    url = f"{datasets_server_url.rstrip('/')}/{path.lstrip('/')}"
    response = requests.get(
        url,
        params=params,
        headers=_build_datasets_server_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _fetch_split_sizes_via_datasets_server(
    dataset_name: str,
    *,
    datasets_server_url: str,
) -> dict[str, int]:
    payload = _fetch_datasets_server_json(
        "size",
        params={"dataset": dataset_name},
        datasets_server_url=datasets_server_url,
    )
    split_sizes: dict[str, int] = {}
    for split_info in payload.get("size", {}).get("splits", []):
        split_name = str(split_info["split"])
        estimated = split_info.get("estimated_num_rows")
        exact = split_info.get("num_rows")
        if estimated is not None:
            split_sizes[split_name] = int(estimated)
        elif exact is not None:
            split_sizes[split_name] = int(exact)
    return split_sizes


def _choose_window_offsets(
    total_rows: int,
    *,
    window_size: int,
    window_count: int,
    max_samples_per_split: int | None,
) -> list[int]:
    if total_rows <= 0:
        return [0]
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_count <= 0:
        raise ValueError("window_count must be positive")

    effective_rows = total_rows
    if max_samples_per_split is not None:
        effective_rows = min(total_rows, max_samples_per_split)

    if effective_rows <= window_size:
        return [0]

    count = min(window_count, max(1, effective_rows // window_size))
    max_offset = max(0, effective_rows - window_size)
    if count == 1 or max_offset == 0:
        return [0]

    offsets = {0, max_offset}
    for index in range(count):
        fraction = index / max(count - 1, 1)
        offsets.add(int(round(max_offset * fraction)))
    return sorted(offsets)


def _fetch_run_ids_window_via_datasets_server(
    dataset_name: str,
    split: str,
    *,
    offset: int,
    length: int,
    datasets_server_url: str,
) -> list[str]:
    payload = _fetch_datasets_server_json(
        "rows",
        params={
            "dataset": dataset_name,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        },
        datasets_server_url=datasets_server_url,
    )
    rows = payload.get("rows", [])
    run_ids: list[str] = []
    for row in rows:
        data = row.get("row", {})
        run_ids.append(str(data.get("run_id", "unknown")))
    return run_ids


def audit_huggingface_split_run_ids_via_datasets_server(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    splits: Sequence[str] = ("train", "validation", "test"),
    datasets_server_url: str = DEFAULT_DATASETS_SERVER_URL,
    max_samples_per_split: int | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_count: int = DEFAULT_WINDOW_COUNT,
) -> dict[str, Any]:
    """Audit splits via lightweight datasets-server row windows.

    This is a bounded evidence audit. It can detect observed overlap, but it does not prove
    global split integrity because it samples windows rather than exhaustively reading rows.
    """

    split_sizes = _fetch_split_sizes_via_datasets_server(
        dataset_name,
        datasets_server_url=datasets_server_url,
    )

    split_samples: dict[str, list[dict[str, str]]] = {}
    observed_offsets: dict[str, list[int]] = {}
    for split in splits:
        total_rows = split_sizes.get(split, 0)
        offsets = _choose_window_offsets(
            total_rows,
            window_size=window_size,
            window_count=window_count,
            max_samples_per_split=max_samples_per_split,
        )
        observed_offsets[split] = offsets

        samples: list[dict[str, str]] = []
        for offset in offsets:
            run_ids = _fetch_run_ids_window_via_datasets_server(
                dataset_name,
                split,
                offset=offset,
                length=window_size,
                datasets_server_url=datasets_server_url,
            )
            samples.extend({"run_id": run_id} for run_id in run_ids)
            if max_samples_per_split is not None and len(samples) >= max_samples_per_split:
                samples = samples[:max_samples_per_split]
                break
        split_samples[split] = samples

    report = audit_split_run_ids(split_samples, max_samples_per_split=max_samples_per_split)
    report["audit_strategy"] = "datasets_server"
    report["is_definitive"] = False
    report["notes"] = (
        "This is a lightweight bounded evidence audit via Hugging Face datasets-server row "
        "windows. It can detect observed overlap but cannot prove global non-overlap."
    )
    report["split_sizes"] = {split: split_sizes.get(split, 0) for split in splits}
    report["window_size"] = window_size
    report["window_count"] = window_count
    report["observed_offsets"] = observed_offsets
    return report


def audit_huggingface_split_run_ids(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    splits: Sequence[str] = ("train", "validation", "test"),
    cache_dir: str | Path | None = None,
    streaming: bool = True,
    max_samples_per_split: int | None = None,
    strategy: str = "datasets_server",
    datasets_server_url: str = DEFAULT_DATASETS_SERVER_URL,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_count: int = DEFAULT_WINDOW_COUNT,
) -> dict[str, Any]:
    """Audit HF split run boundaries with a lightweight API-first strategy."""

    if strategy == "datasets_server":
        return audit_huggingface_split_run_ids_via_datasets_server(
            dataset_name=dataset_name,
            splits=splits,
            datasets_server_url=datasets_server_url,
            max_samples_per_split=max_samples_per_split,
            window_size=window_size,
            window_count=window_count,
        )
    if strategy != "datasets":
        raise ValueError("strategy must be 'datasets_server' or 'datasets'")

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
    report["audit_strategy"] = "datasets"
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
        choices=("datasets_server", "datasets"),
        default="datasets_server",
        help="Split audit backend. datasets_server is lightweight and provisional; datasets is heavier.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Rows to request per datasets-server sampling window.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=DEFAULT_WINDOW_COUNT,
        help="Number of datasets-server sampling windows per split.",
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
        window_size=args.window_size,
        window_count=args.window_count,
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
