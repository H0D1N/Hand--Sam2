import torch
from torch import nn

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones

class MultiViewDistributionLayer(nn.Module):
    """
    使用共享特征更新每个视角的空间特征。

    Cross Attention:
        Q:   view_features + view_pos
        K/V: shared_features
    """
    def __init__(
        self,
        d_model: int,
        cross_attention: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()

        # 1. Cross Attention：
        # latent读取所有视角的图像token。
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.cross_attn = cross_attention
        self.cross_attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        view_tokens: torch.Tensor,    # [B, V*N, C]
        view_pos: torch.Tensor,       # [B, V*N, C]
        shared_tokens: torch.Tensor,  # [B, M, C]
    ) -> torch.Tensor:
        view_tokens_norm = self.cross_attention_norm(view_tokens)

        delta = self.cross_attn(
            q=view_tokens_norm + view_pos,
            k=shared_tokens,
            v=shared_tokens,
        )

        return view_tokens + self.cross_attention_dropout(delta)


class MultiViewFeatureDistributor(nn.Module):
    """
    将共享特征 [B, M, C] 分发回各视角 [B, V, N, C]。
    """

    def __init__(self, d_model, layer, num_layers):
        super().__init__()

        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        view_features: torch.Tensor,    # [B, V, N, C]
        view_pos: torch.Tensor,         # [B, V, N, C]
        shared_tokens: torch.Tensor,    # [B, M, C]
    ) -> torch.Tensor:
        assert view_features.ndim == 4
        assert shared_tokens.ndim == 3
        assert view_pos.shape == view_features.shape

        B, V, N, C = view_features.shape

        assert shared_tokens.shape[0] == B
        assert shared_tokens.shape[2] == C

        # [B,V,N,C] -> [B,V*N,C]
        view_tokens = view_features.reshape(B, V * N, C)
        view_token_pos = view_pos.reshape(B, V * N, C)

        for layer in self.layers:
            view_tokens = layer(
                view_tokens=view_tokens,
                view_pos=view_token_pos,
                shared_tokens=shared_tokens,
            )

        # [B,V*N,C] -> [B,V,N,C]
        distributed_features = self.norm(view_tokens).reshape(B, V, N, C)

        return distributed_features
