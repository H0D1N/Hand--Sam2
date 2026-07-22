from __future__ import annotations

import types
from collections.abc import Iterator

import torch
from torch import nn

class ResidualAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        bottleneck_dim = max(1, min(hidden_dim, adapter_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.scale = nn.Parameter(torch.full((1,), init_scale))

        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        residual = self.down_proj(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.up_proj(residual)
        return x + self.scale * residual


def inject_image_encoder_adapters(
    model: nn.Module,
    adapter_dim: int = 64,
    adapter_dropout: float = 0.1,
    adapter_init_scale: float = 1e-3,
) -> int:
    trunk = model.image_encoder.trunk
    injected = 0
    for block in trunk.blocks:
        if getattr(block, "_adapter_injected", False):
            continue

        reference_parameter = next(block.parameters(), None)
        adapter = ResidualAdapter(
            hidden_dim=block.dim_out,
            adapter_dim=adapter_dim,
            dropout=adapter_dropout,
            init_scale=adapter_init_scale,
        )
        if reference_parameter is not None:
            adapter = adapter.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        block.adapter = adapter
        block._forward_without_adapter = block.forward

        def _forward_with_adapter(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            x = self._forward_without_adapter(x)
            return self.adapter(x)

        block.forward = types.MethodType(_forward_with_adapter, block)
        block._adapter_injected = True
        injected += 1

    return injected

def iter_image_encoder_adapters(
    model: nn.Module,
) -> Iterator[ResidualAdapter]:
    for block in model.image_encoder.trunk.blocks:
        adapter = getattr(block, "adapter", None)

        if adapter is not None:
            yield adapter
