import torch
from torch import nn

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones

class MultiViewAggregationLayer(nn.Module):
    """
    使用 M 个 latent token 聚合同一时刻的 V 个视角

    顺序：
        Cross Attention
        -> Self Attention
        -> FFN
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        self_attention: nn.Module,
        cross_attention: nn.Module,
    ) -> None:
        super().__init__()

        # 1. Cross Attention：
        # latent读取所有视角的图像token。
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.cross_attn = cross_attention
        self.cross_attention_dropout = nn.Dropout(dropout)

        # 2. Self Attention：
        # 聚合后的latent互相交流。
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.self_attn = self_attention
        self.self_attention_dropout = nn.Dropout(dropout)


        # 3. FFN：
        # 保留SAM2 MemoryAttentionLayer的单个FFN。
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_linear2 = nn.Linear(dim_feedforward, d_model)

        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tgt, source_tokens, source_pos):
        tgt = self._forward_cross_attention(tgt, source_tokens, source_pos)
        tgt = self._forward_self_attention(tgt)
        tgt = self._forward_mlp(tgt)
        return tgt

    def _forward_cross_attention(self, tgt, source_tokens, source_pos):
        tgt_res = self.cross_attention_norm(tgt)

        tgt_res = self.cross_attn(
            q=tgt_res,
            k=source_tokens + source_pos,
            v=source_tokens,
        )

        return tgt + self.cross_attention_dropout(tgt_res)

    def _forward_self_attention(self, tgt):
        tgt_res = self.self_attention_norm(tgt)

        tgt_res = self.self_attn(
            q=tgt_res,
            k=tgt_res,
            v=tgt_res,
        )

        return tgt + self.self_attention_dropout(tgt_res)

    def _forward_mlp(self, tgt):
        tgt_res = self.ffn_norm(tgt)
        tgt_res = self.ffn_linear1(tgt_res)
        tgt_res = self.activation(tgt_res)
        tgt_res = self.ffn_dropout(tgt_res)
        tgt_res = self.ffn_linear2(tgt_res)
        tgt_res = self.output_dropout(tgt_res)

        return tgt + tgt_res
    
class MultiViewFeatureAggregator(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer: nn.Module,
        num_layers: int,
        num_latents: int,
    ):
        """
        相较 MemoryAttn 少了
            batch_first: forward输入为 BVNC
            pos_enc_at_input: forward输出的 BMC 与位置无关
        """
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # M个聚合Query
        self.latent_tokens = nn.Parameter(
            torch.empty(1, num_latents, d_model)
        )
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)


    def forward(
        self,
        multiview_features: torch.Tensor,  # [B, V, N, C]
        multiview_pos: torch.Tensor,  # [B, V, N, C] pos_enc for multiview_features
    ):
        assert multiview_features.ndim == 4
        assert multiview_pos.shape == multiview_features.shape

        B, V, N, C = multiview_features.shape
        assert C == self.d_model

        view_tokens = multiview_features.reshape(B, V * N, C)
        view_token_encoding = multiview_pos.reshape(B, V * N, C)

        shared_tokens = self.latent_tokens.expand(B, -1, -1)

        for layer in self.layers:
            shared_tokens = layer(
                tgt=shared_tokens,
                source_tokens=view_tokens,
                source_pos=view_token_encoding,
            )

        return self.norm(shared_tokens)
