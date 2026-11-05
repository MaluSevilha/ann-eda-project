"""Definição configurável da MLP de regressão."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    """Arquitetura e regularização estrutural da MLP."""

    input_dim: int = 420
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    normalization: Literal["none", "layer"] = "none"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim deve ser positivo")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims deve conter larguras positivas")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar no intervalo [0, 1)")
        if self.normalization not in {"none", "layer"}:
            raise ValueError("normalization deve ser 'none' ou 'layer'")


class MLPRegressor(nn.Module):
    """MLP com ReLU nas ocultas e uma única saída linear."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        previous = config.input_dim
        for width in config.hidden_dims:
            layers.append(nn.Linear(previous, width))
            if config.normalization == "layer":
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        linear_layers = [
            layer for layer in self.network if isinstance(layer, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        output = linear_layers[-1]
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def count_trainable_parameters(model: nn.Module) -> int:
    """Conta apenas parâmetros que recebem gradientes."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
