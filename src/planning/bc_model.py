"""Camera-only ResNet-18 plus temporal Transformer behavioral cloning model."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class BehavioralCloningModel(nn.Module):
    """Predict ``[steering, throttle]`` from four past/current RGB frames."""

    def __init__(
        self,
        *,
        sequence_length: int = 4,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.0,
        pretrained: bool = False,
        freeze_early_layers: bool = True,
    ) -> None:
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if 512 % transformer_heads != 0:
            raise ValueError("transformer_heads must divide the 512-dimensional feature size")
        self.sequence_length = sequence_length
        self.feature_dim = 512
        self.backbone = self._build_backbone(pretrained)
        if freeze_early_layers:
            backbone = cast(Any, self.backbone)
            for layer in (backbone.layer1, backbone.layer2):
                for parameter in layer.parameters():
                    parameter.requires_grad = False
        self.position = nn.Parameter(torch.zeros(1, sequence_length, self.feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=transformer_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Linear(128, 2))

    @staticmethod
    def _build_backbone(pretrained: bool) -> nn.Module:
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torchvision is required for the BC model") from exc
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        return model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(video.shape)}")
        batch, time, channels, height, width = video.shape
        if time != self.sequence_length or channels != 3:
            raise ValueError(
                f"Expected time={self.sequence_length}, channels=3; got {time}, {channels}"
            )
        features = self.backbone(video.reshape(batch * time, channels, height, width))
        features = features.reshape(batch, time, self.feature_dim)
        encoded = self.temporal_encoder(features + self.position)
        output = self.head(encoded[:, -1])
        if output.shape != (batch, 2):
            raise RuntimeError(f"Expected output shape {(batch, 2)}, got {tuple(output.shape)}")
        return output


def build_bc_model(config: dict[str, Any]) -> BehavioralCloningModel:
    """Construct a BC model from a plain configuration mapping."""
    return BehavioralCloningModel(
        sequence_length=int(config.get("sequence_length", 4)),
        transformer_heads=int(config.get("transformer_heads", 4)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
        pretrained=bool(config.get("pretrained", False)),
        freeze_early_layers=bool(config.get("freeze_early_layers", True)),
    )
