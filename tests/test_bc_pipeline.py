from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.data.bc_sequences import BehavioralCloningSequenceDataset, collate_bc_sequences
from src.planning.bc_evaluation import action_metrics
from src.planning.bc_training import load_checkpoint, save_checkpoint, weighted_bc_loss
from src.transforms.bc_augmentation import shift_sequence_horizontally


def _sample(run_id: str, frame: int, steer: float = 0.1) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "frame": frame,
        "timestamp": frame / 20.0,
        "speed_kmh": 20.0,
        "image_front": np.full((6, 8, 3), frame, dtype=np.uint8),
        "seg_front": np.zeros((6, 8), dtype=np.uint8),
        "lidar": np.array([[4.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "throttle": 0.4,
        "steer": steer,
        "brake": 0.0,
    }


def test_sequence_windows_do_not_cross_runs_and_reorder_target() -> None:
    samples = [_sample("a", frame, steer=0.2 if frame == 3 else 0.1) for frame in range(4)]
    samples += [_sample("b", frame) for frame in range(2)]
    dataset = BehavioralCloningSequenceDataset(samples, sequence_length=4)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["rgb"].shape == (4, 3, 6, 8)
    assert item["bc_target"].tolist() == pytest.approx([0.2, 0.4])
    assert item["hard_control"] is True
    assert item["metadata"]["temporal_horizon_seconds"] == pytest.approx(0.15)


def test_sequence_windows_drop_frame_id_discontinuities() -> None:
    samples = [_sample("a", frame) for frame in (0, 5, 10, 20)]
    assert len(BehavioralCloningSequenceDataset(samples, sequence_length=4)) == 0


def test_sequence_collate_stacks_targets_and_weights() -> None:
    samples = [_sample("a", frame) for frame in range(4)]
    item = BehavioralCloningSequenceDataset(samples)[0]
    batch = collate_bc_sequences([item, item])
    assert batch["rgb"].shape == (2, 4, 3, 6, 8)
    assert batch["bc_target"].shape == (2, 2)
    assert batch["sample_weight"].shape == (2,)


def test_horizontal_shift_returns_corrective_offset() -> None:
    frames = torch.ones((4, 3, 4, 8))
    shifted, correction = shift_sequence_horizontally(frames, shift_pixels=2)
    assert shifted.shape == frames.shape
    assert correction == pytest.approx(-0.004)
    assert torch.count_nonzero(shifted[..., :2]) == 0


def test_weighted_loss_and_action_metrics_use_steer_then_throttle_order() -> None:
    prediction = torch.tensor([[0.2, 0.5], [0.0, 0.4]])
    target = torch.tensor([[0.1, 0.4], [0.0, 0.5]])
    loss, metrics = weighted_bc_loss(prediction, target, steering_weight=2.0)
    assert float(loss) > 0.0
    assert metrics["steering_loss"] == pytest.approx(0.005)
    result = action_metrics(prediction, target)
    assert result["steering_mae"] == pytest.approx(0.05)
    assert result["throttle_mae"] == pytest.approx(0.1)


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        best_val_loss=0.25,
        config={"sequence_length": 4},
    )
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer)
    assert state["epoch"] == 3
    assert state["best_val_loss"] == pytest.approx(0.25)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, source[name])


def test_model_shape_contract_if_torchvision_is_available() -> None:
    pytest.importorskip("torchvision")
    from src.planning.bc_model import BehavioralCloningModel

    model = BehavioralCloningModel(sequence_length=4, pretrained=False)
    output = model(torch.randn(1, 4, 3, 64, 64))
    assert output.shape == (1, 2)
