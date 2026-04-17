# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import Attention, Block, LayerScale, Mlp
from tokengs.options import Options


class DecoderBlock(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            qk_norm: bool,
            flex_attn_score_mod=None,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fused_attn=True,
                flex_attn_score_mod=flex_attn_score_mod,
            )

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            queries_normed = self.norm(gs_tokens)
            return self.gs_self_attn(queries_normed)

    class CrossAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            q_norm: bool,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)

        def forward(
            self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
        ) -> torch.Tensor:
            gs_tokens_normed = self.gs_token_norm(gs_tokens)
            q = rearrange(self.q_proj(gs_tokens_normed), "b n (h d) -> b h n d", h=self.num_heads)
            q = self.q_norm(q)

            cross_attn_output = F.scaled_dot_product_attention(q, keys, values)
            cross_attn_output = rearrange(cross_attn_output, "b h n d -> b n (h d)")
            return self.out_proj(cross_attn_output)

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
        attn_score_mod=None,
    ):
        super().__init__()

        def make_scale() -> nn.Module:
            return LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.gs_self_attn = DecoderBlock.SelfAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            qk_norm=qk_norm,
            flex_attn_score_mod=attn_score_mod,
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            q_norm=qk_norm,
        )
        self.gs_cross_attn_scale = make_scale()

        self.mlp = DecoderBlock.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

    def forward(
        self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        gs_tokens = gs_tokens + self.gs_cross_attn_scale(self.gs_cross_attn(gs_tokens, keys, values))
        gs_tokens = gs_tokens + self.gs_self_attn_scale(self.gs_self_attn(gs_tokens))
        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))

        return gs_tokens


class EncDecBackbone(nn.Module):
    """
    An encoder-decoder backbone for the EncDec architecture.

    Encoder: a stack of ViT blocks which produce a latent representation. This is followed by a key-value projection to produce a key and value for the decoder.

    Decoder: a stack of transformer decoder layers which attend from GS tokens to the encoder output and among themselves.
    """

    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.encoder = nn.Sequential(
            *[
                Block(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=None,
                    flex_attn_block_mask=None,
                )
                for _ in range(self.opt.enc_depth)
            ]
        )

        self.encoder_norm = nn.LayerNorm(self.opt.enc_embed_dim)

        self.kv_proj = nn.Linear(
            self.opt.enc_embed_dim, self.opt.enc_embed_dim * 2, bias=True
        )

        # normalize the k projection, q are normalized in the attention block
        self.k_proj_norm = nn.LayerNorm(self.opt.enc_embed_dim // self.opt.enc_num_heads)

        if self.opt.num_dynamic_gs_tokens > 0:
            block_ids = [0] * self.opt.num_gs_tokens + [1] * self.opt.num_dynamic_gs_tokens
            self.register_buffer(
                "_decoder_block_ids",
                torch.tensor(block_ids, dtype=torch.long),
                persistent=False,
            )

            def block_causal_score_mod(score, b, h, q_idx, kv_idx, block_ids_tensor: torch.Tensor):
                same_block_mask = block_ids_tensor[q_idx] == block_ids_tensor[kv_idx]
                causal_mask = q_idx >= kv_idx
                return torch.where(same_block_mask | causal_mask, score, float("-inf"))

            score_mod = partial(block_causal_score_mod, block_ids_tensor=self._decoder_block_ids)
        else:
            score_mod = None

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=5e-3 * self.opt.gs_token_std,
                    qk_norm=True,
                    attn_score_mod=score_mod,
                )
                for _ in range(self.opt.dec_depth)
            ]
        )

    def _encode_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encoder(image_features)
        image_features = self.encoder_norm(image_features)
        image_feature_keys, image_feature_values = rearrange(
            self.kv_proj(image_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        image_feature_keys = self.k_proj_norm(image_feature_keys)
        return image_feature_keys, image_feature_values
