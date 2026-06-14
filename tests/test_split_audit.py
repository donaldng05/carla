from __future__ import annotations

from typing import Any

import pytest

from src.data.split_audit import (
    audit_huggingface_split_run_ids,
    format_split_audit_report,
    main,
)


def test_format_split_audit_report_mentions_overlap_and_split_summary() -> None:
    report = {
        "has_overlap": True,
        "overlaps": {"run_008": ["train", "validation"]},
        "audit_strategy": "datasets_server",
        "is_definitive": False,
        "notes": "bounded evidence audit",
        "window_size": 25,
        "window_count": 5,
        "split_sizes": {"train": 60000, "validation": 8000, "test": 7000},
        "observed_offsets": {"train": [0, 100], "validation": [0, 50], "test": [0, 50]},
        "runs_by_split": {
            "train": ["run_008", "run_020"],
            "validation": ["run_001", "run_008"],
            "test": ["run_030"],
        },
        "run_counts": {"train": 2, "validation": 2, "test": 1},
        "inspected_counts": {"train": 100, "validation": 50, "test": 25},
    }

    text = format_split_audit_report(
        report,
        dataset_name="demo-dataset",
        splits=("train", "validation", "test"),
        streaming=True,
        max_samples_per_split=100,
    )

    assert "Dataset: demo-dataset" in text
    assert "Audit strategy: datasets_server" in text
    assert "Definitive audit: False" in text
    assert "Run overlap detected: True" in text
    assert "- train: inspected=100, unique_runs=2" in text
    assert "- run_008: train, validation" in text


def test_main_returns_zero_when_no_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data.split_audit.audit_huggingface_split_run_ids",
        lambda **_: {
            "has_overlap": False,
            "overlaps": {},
            "runs_by_split": {"train": ["run_020"], "validation": ["run_008"], "test": ["run_001"]},
            "run_counts": {"train": 1, "validation": 1, "test": 1},
            "inspected_counts": {"train": 10, "validation": 10, "test": 10},
        },
    )

    exit_code = main(["--max-samples-per-split", "10"])

    assert exit_code == 0


def test_main_returns_one_when_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data.split_audit.audit_huggingface_split_run_ids",
        lambda **_: {
            "has_overlap": True,
            "overlaps": {"run_008": ["train", "validation"]},
            "runs_by_split": {"train": ["run_008"], "validation": ["run_008"], "test": ["run_001"]},
            "run_counts": {"train": 1, "validation": 1, "test": 1},
            "inspected_counts": {"train": 10, "validation": 10, "test": 10},
        },
    )

    exit_code = main(["--max-samples-per-split", "10", "--json"])

    assert exit_code == 1


def test_audit_huggingface_split_run_ids_uses_datasets_server_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.data.split_audit.audit_huggingface_split_run_ids_via_datasets_server",
        lambda **_: {
            "has_overlap": False,
            "overlaps": {},
            "audit_strategy": "datasets_server",
            "is_definitive": False,
            "notes": "bounded evidence audit",
            "runs_by_split": {"train": ["run_020"], "validation": ["run_008"], "test": ["run_001"]},
            "run_counts": {"train": 1, "validation": 1, "test": 1},
            "inspected_counts": {"train": 25, "validation": 25, "test": 25},
        },
    )

    report = audit_huggingface_split_run_ids()

    assert report["audit_strategy"] == "datasets_server"
    assert report["is_definitive"] is False
