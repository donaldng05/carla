"""PyTorch data loading and preprocessing helpers for CARLA multimodal samples."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_DATASET_NAME",
    "CarlaDataPreprocessingConfig",
    "CarlaMultimodalDataset",
    "audit_huggingface_split_run_ids",
    "audit_split_run_ids",
    "build_bc_preprocessing_config",
    "build_carla_dataset_from_hf",
    "build_occupancy_preprocessing_config",
    "collate_carla_samples",
    "format_split_audit_report",
    "validate_processed_dataset",
]

_SPLIT_AUDIT_EXPORTS = {
    "DEFAULT_DATASET_NAME",
    "audit_huggingface_split_run_ids",
    "audit_split_run_ids",
    "format_split_audit_report",
}

_CARLA_DATASET_EXPORTS = {
    "CarlaDataPreprocessingConfig",
    "CarlaMultimodalDataset",
    "build_bc_preprocessing_config",
    "build_carla_dataset_from_hf",
    "build_occupancy_preprocessing_config",
    "collate_carla_samples",
    "validate_processed_dataset",
}


def __getattr__(name: str) -> Any:
    if name in _SPLIT_AUDIT_EXPORTS:
        module = import_module("src.data.split_audit")
        return getattr(module, name)

    if name in _CARLA_DATASET_EXPORTS:
        module = import_module("src.data.carla_dataset")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
