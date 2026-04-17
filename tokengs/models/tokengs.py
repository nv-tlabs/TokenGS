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

"""
TokenGS: Encoder-decoder model for 3D scene reconstruction from sparse views.
"""

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from lpips import LPIPS

from tokengs.options import Options
from tokengs.models.input_types import (
    EncoderLatent,
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    split_data,
)
from .attention import PatchEmbed
from tokengs.models.enc_dec import EncDecBackbone
from tokengs.rendering.gs import GaussianRenderer
from tokengs.models.activations import ClipActivationHead
from tokengs.models.losses import compute_tokengs_loss
from tokengs.utils.training import freeze_model_parameters


class TokenGS(nn.Module):
    """
    TokenGS model with encoder-decoder architecture.
    
    Uses separate encoder and decoder with cross-attention for Gaussian token processing.
    """
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        
        self.img_size = self.opt.img_size if not isinstance(self.opt.img_size, int) else [self.opt.img_size, self.opt.img_size]

        norm_layer_factory = partial(nn.LayerNorm, bias=True)

        self.patch_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=3,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        self.patch_plucker_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=6,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        if self.opt.time_embedding:
            self.patch_time_embed = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )
            self.patch_time_embed_tgt = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )

            for m in (self.patch_time_embed, self.patch_time_embed_tgt):
                m.proj.weight.data.fill_(0.0)
                m.proj.bias.data.fill_(0.0)

        # Encoder-decoder architecture
        self.enc_dec_backbone = EncDecBackbone(opt)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activation head (always clip)
        self.activation_head = ClipActivationHead(opt)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # Learnable GS tokens
        self.gs_tokens = nn.Parameter(
            self.opt.gs_token_std * torch.randn(self.opt.num_gs_tokens, self.opt.token_dim)
        )

        if self.opt.num_dynamic_gs_tokens > 0:
            self.gs_tokens_dynamic = nn.Parameter(
                self.opt.gs_token_std * torch.randn(self.opt.num_dynamic_gs_tokens, self.opt.token_dim)
            )

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k:
                del state_dict[k]
        return state_dict

    def _background_color(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.opt.bg_color == "white":
            return torch.ones(3, dtype=dtype, device=device)
        if self.opt.bg_color == "black":
            return torch.zeros(3, dtype=dtype, device=device)
        if self.opt.bg_color == "grey":
            return torch.ones(3, dtype=dtype, device=device) * 0.5
        raise ValueError(f"Invalid background color: {self.opt.bg_color}")

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        """
        Encode input views into latent representation (keys and values for cross-attention).
        
        Args:
            encoder_input: ModelInputEncoder containing input view data (only input views)
            
        Returns:
            EncoderLatent containing keys and values for cross-attention
        """
        B, V, _, H, W = encoder_input.images_rgb.shape
        height = int(H // self.opt.patch_size)
        width = int(W // self.opt.patch_size)

        assert height * self.opt.patch_size == H, f"H={H} must be divisible by patch_size={self.opt.patch_size}"
        assert width * self.opt.patch_size == W, f"W={W} must be divisible by patch_size={self.opt.patch_size}"

        # Reshape for tokenization
        images_rgb_reshaped = encoder_input.images_rgb.reshape(B * V, 3, H, W)
        plucker_reshaped = encoder_input.plucker.reshape(B * V, 6, H, W)
        
        # Embed RGB patches
        x = self.patch_embed(images_rgb_reshaped)
        # Add Plucker embeddings
        x_plucker_emb = self.patch_plucker_embed(plucker_reshaped)
        x = x + x_plucker_emb  # B*V, N, C

        # Reshape from (B*V, N, C) to (B, V*N, C) so the encoder sees one sequence per batch
        x = rearrange(x, "(b v) n c -> b (v n) c", b=B, v=V)

        # Add time embeddings if enabled (processed in 2D image space, then reshaped)
        if self.opt.time_embedding:
            time_emb_reshaped = encoder_input.time_embedding_input.reshape(B * V, self.opt.time_embedding_dim, H, W)
            x_time_emb = self.patch_time_embed(time_emb_reshaped)  # (B*V, N, C)
            x = x + x_time_emb.reshape(B, V * (height * width), -1)  # Reshape to match x: (B, V*N, C)

        # Run encoder on joint views and produce keys/values
        # x shape: (B, V*N, C) where all views are processed jointly
        image_feature_keys, image_feature_values = self.enc_dec_backbone._encode_to_kv(x)
        
        return EncoderLatent(
            keys=image_feature_keys,
            values=image_feature_values,
        )

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        """
        Get initial GS tokens (without time conditioning).
        Useful for test-time training where you want to optimize the GS tokens.
        Time conditioning is applied later in forward_decoder.
        
        Args:
            batch_size: Batch size
            
        Returns:
            GS tokens with shape [B, num_gs_tokens, C]
        """
        batch_gs_tokens = self.gs_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.opt.num_dynamic_gs_tokens > 0:
            dyn = self.gs_tokens_dynamic.unsqueeze(0).repeat(batch_size, 1, 1)
            batch_gs_tokens = torch.cat([batch_gs_tokens, dyn], dim=1)
        return batch_gs_tokens
    
    def _apply_time_embedding_to_gs_tokens(
        self, 
        gs_tokens: torch.Tensor,
        decoder_input: ModelInputDecoder,
    ) -> torch.Tensor:
        """
        Apply target time embedding to GS tokens.
        This is called by forward_decoder to condition GS tokens on target time.
        
        Args:
            gs_tokens: Base GS tokens [B, num_gs_tokens, C]
            decoder_input: ModelInputDecoder containing target time embedding
            
        Returns:
            Time-conditioned GS tokens [B, num_gs_tokens (+dynamic), C]
        """
        if not self.opt.time_embedding or decoder_input.time_embedding_target is None:
            # No time embedding or no target time provided
            return gs_tokens
        
        B = gs_tokens.shape[0]
        _, _, T_dim, H_tgt, W_tgt = decoder_input.time_embedding_target.shape
        
        # Process target time embedding through patch embedding
        time_emb_target = decoder_input.time_embedding_target.reshape(B, T_dim, H_tgt, W_tgt)
        x_time_emb_tgt = self.patch_time_embed_tgt(time_emb_target)
        
        if self.opt.num_dynamic_gs_tokens > 0:
            # For dynamic tokens: add time embedding only to dynamic tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[:, :self.opt.num_dynamic_gs_tokens, :]
            gs_tokens[:, -self.opt.num_dynamic_gs_tokens:] = gs_tokens[:, -self.opt.num_dynamic_gs_tokens:] + x_time_emb_tgt
        else:
            # For standard learnable tokens: add time embedding to all GS tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[:, :self.opt.num_gs_tokens, :]
            gs_tokens = gs_tokens + x_time_emb_tgt
        
        return gs_tokens

    def forward_decoder(
        self, 
        encoder_latent: EncoderLatent,
        decoder_input: ModelInputDecoder,
        gs_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process latent representation (keys/values) to Gaussians using decoder with cross-attention.
        
        Args:
            encoder_latent: EncoderLatent containing keys and values from the encoder
            decoder_input: ModelInputDecoder containing rendering parameters and target time
            gs_tokens: Optional precomputed GS tokens [B, num_gs_tokens, C]. If None, creates new ones.
            
        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        B = encoder_latent.keys.shape[0]
        
        # Get or use provided GS tokens (without time conditioning)
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=B)

        # Apply time embedding to GS tokens
        gs_tokens = self._apply_time_embedding_to_gs_tokens(gs_tokens, decoder_input)

        # Run decoder with precomputed keys/values
        for layer in self.enc_dec_backbone.decoder_blocks:
            gs_tokens = layer(
                gs_tokens=gs_tokens,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )

        # Convert to Gaussians
        gaussians = self.activation_head(gs_tokens)

        # give gaussians an offset to the z axis so it is visible when initialized
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset

        return gaussians

    def forward_gaussians(self, model_input: ModelInput) -> torch.Tensor:
        """
        Generate Gaussians from model input.
        This is a convenience method that combines encoding and decoding.
        For test-time training, use forward_encoder(), get_gs_tokens(), and forward_decoder() separately.
        
        Args:
            model_input: ModelInput containing all necessary input data
            
        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Encode input views
        encoder_latent = self.forward_encoder(model_input.encoder)
        
        # Decode to Gaussians (time conditioning is applied inside forward_decoder)
        gaussians = self.forward_decoder(encoder_latent, model_input.decoder)
        
        return gaussians

    def render_gaussians(self, gaussians: torch.Tensor, decoder_input: ModelInputDecoder) -> dict:
        """
        Render Gaussians to images.
        
        Args:
            gaussians: Gaussian parameters [B, N, 14]
            decoder_input: ModelInputDecoder containing camera parameters
            
        Returns:
            Dictionary with 'images_pred', 'alphas_pred', etc.
        """
        bg_color = self._background_color(gaussians.dtype, gaussians.device)
        results = self.gs.render(
            gaussians,
            decoder_input.cam_view,
            bg_color=bg_color,
            intrinsics=decoder_input.intrinsics,
        )
        return results

    def compute_loss(
        self,
        gaussians: torch.Tensor,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> dict:
        render_results = self.render_gaussians(gaussians, decoder_input)
        return compute_tokengs_loss(
            opt=self.opt,
            img_size=self.img_size,
            render_results=render_results,
            supervision=supervision,
            gaussians_xyz=gaussians[..., :3],
            lpips_loss=getattr(self, "lpips_loss", None),
        )

    def forward(self, data, skip_loss=False):
        """
        Forward pass of the TokenGS model.
        
        Args:
            data: Dictionary from dataloader
            skip_loss: If True, skip loss computation
            
        Returns:
            Dictionary containing results including 'loss', 'gaussians', 'images_pred', etc.
        """
        # Split data into structured input and supervision
        model_input, supervision = split_data(data, self.opt)

        # Generate Gaussians from input
        if self.opt.use_ttt_for_eval:
            gaussians = self.forward_ttt(model_input, n_steps=self.opt.ttt_n_steps, lr=self.opt.ttt_lr)  # [B, N, 14]
        else:
            gaussians = self.forward_gaussians(model_input)  # [B, N, 14]

        # Compute loss or just render
        if skip_loss:
            results = self.render_gaussians(gaussians, model_input.decoder)
            results['gaussians'] = gaussians
            return results
        
        # Compute loss
        results = self.compute_loss(gaussians, model_input.decoder, supervision)
        results['gaussians'] = gaussians
        results['images_output'] = supervision.images_output
        return results

    def forward_ttt(
        self,
        model_input: ModelInput,
        n_steps: int = 10,
        lr: float = 1e-3,
    ) -> torch.Tensor:
        """Test-time training: optimize Gaussian tokens for `n_steps` against input views."""
        return self.forward_ttt_tokens(model_input, n_steps, lr)

    def forward_ttt_tokens(self, model_input: ModelInput, n_steps: int = 10, lr: float = 1e-3) -> torch.Tensor:
        """
        Forward pass of the TokenGS model using test-time training of the Gaussian tokens for `n_steps`.
        
        This method optimizes the Gaussian tokens by rendering to input views and 
        comparing against the input view images as supervision.

        Args:
            model_input: ModelInput containing encoder and decoder inputs (same signature as forward_gaussians)
            n_steps: Number of test-time training steps
            lr: Learning rate for the optimizer

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Convert to TTT inputs (render to input views, supervise with input images)
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        
        with freeze_model_parameters(self):
            return self._forward_ttt_tokens(model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _forward_ttt_tokens(
        self, model_input: ModelInput, supervision: ModelSupervision, n_steps: int = 10, lr: float = 1e-3,
    ) -> torch.Tensor:
        """Inner loop for forward_ttt_tokens; caller holds model frozen."""
        encoder_latent = self.forward_encoder(model_input.encoder)

        with torch.set_grad_enabled(True):
            gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach().requires_grad_(True)
            optim = torch.optim.Adam([gs_tokens], lr=lr)
            for _ in range(n_steps):
                optim.zero_grad()
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                loss = self.compute_loss(gaussians, model_input.decoder, supervision)["loss"]
                loss.backward()
                optim.step()
        
        # compute the final result
        return self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens.detach())
