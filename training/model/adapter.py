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

def inject_mask_decoder_adapters(
    model: nn.Module,
    adapter_dim: int = 64,
    adapter_dropout: float = 0.1,
    adapter_init_scale: float = 1e-3,
) -> int:
    """
    在左右 MaskDecoder 的每个 TwoWayAttentionBlock 后，
    向 query token 注入独立的 ResidualAdapter。

    SAM2 Tiny 每个 MaskDecoder 有两个 Transformer block，
    因此左右两个 decoder 一共注入四个 Adapter。
    """
    injected = 0

    for decoder in (model.left_mask_decoder, model.right_mask_decoder):
        hidden_dim = decoder.transformer.embedding_dim

        for layer in decoder.transformer.layers:
            if getattr(layer, "_decoder_adapter_injected", False):
                continue

            reference_parameter = next(layer.parameters(), None)
            adapter = ResidualAdapter(
                hidden_dim=hidden_dim,
                adapter_dim=adapter_dim,
                dropout=adapter_dropout,
                init_scale=adapter_init_scale,
            )

            if reference_parameter is not None:
                adapter = adapter.to(device=reference_parameter.device, dtype=reference_parameter.dtype)

            # 赋值给 layer 后，Adapter 会自动注册进 state_dict。
            layer.decoder_adapter = adapter

            # 保存原始 TwoWayAttentionBlock.forward。
            layer._forward_without_decoder_adapter = layer.forward

            def _forward_with_decoder_adapter(
                self: nn.Module,
                queries: torch.Tensor,
                keys: torch.Tensor,
                query_pe: torch.Tensor,
                key_pe: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                queries, keys = (
                    self._forward_without_decoder_adapter(
                        queries=queries,
                        keys=keys,
                        query_pe=query_pe,
                        key_pe=key_pe,
                    )
                )

                # 只适配少量 query token。
                queries = self.decoder_adapter(queries)

                return queries, keys

            layer.forward = types.MethodType(_forward_with_decoder_adapter, layer)

            layer._decoder_adapter_injected = True
            injected += 1

    return injected

def iter_mask_decoder_adapters(
    model: nn.Module,
) -> Iterator[ResidualAdapter]:
    for decoder in (model.left_mask_decoder, model.right_mask_decoder):
        for layer in decoder.transformer.layers:
            adapter = getattr(layer, "decoder_adapter", None)

            if adapter is not None:
                yield adapter
