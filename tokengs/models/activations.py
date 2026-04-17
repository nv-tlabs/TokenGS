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

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from tokengs.options import Options


class BaseActivationHead(nn.Module, ABC):
    """
    Base class for converting tokens to Gaussians.

    This is a minimal interface - subclasses can have completely different
    internal structures and initialization logic.
    """

    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt

        self.num_gaussians_per_token = self.opt.patch_size**2

        scale_cap = self.opt.gaussian_scale_cap
        self.scale_shift = 1 - math.log(scale_cap)
        self.scale_cap = scale_cap

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians.

        Args:
            x: Input tokens [B, N, C] where N is number of tokens
            rays_os: Optional ray origins [B, V, 3, H, W] for depth prediction
            rays_ds: Optional ray directions [B, V, 3, H, W] for depth prediction

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        pass


class ClipActivationHead(BaseActivationHead):
    """
    Clip-based activation head that uses exponential activations with hard clipping.
    Uses a single deconv (linear) layer for all Gaussian components.
    Supports both xyz and depth prediction modes.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

        # Set output dimensions based on prediction mode
        self.output_dims = (
            3 + 1 + 3 + 4 + 3
        )  # x, y, z, opacity, scale, rotation, rgb

        # Single linear layer (deconv) for all components
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.opt.patch_size * self.opt.patch_size,
            bias=True,
        )

        self._init_weights(self.deconv)

        # Mark parameters to exclude from weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.002)
            if m.bias is not None:
                m.bias.data.zero_()

    def pos_act(self, x: torch.Tensor) -> torch.Tensor:
        """Position activation without scaling factor (for xyz prediction)"""
        pos = torch.sign(x) * (torch.expm1(torch.abs(x)))  # inverse log transform
        return pos

    def scale_act(self, x: torch.Tensor) -> torch.Tensor:
        """Scale activation with exponential and hard clipping"""
        x = x - self.scale_shift
        return torch.minimum(
            torch.exp(x),
            torch.tensor(self.scale_cap, device=x.device, dtype=x.dtype),
        )

    def opacity_act(self, x: torch.Tensor) -> torch.Tensor:
        """Opacity activation with hard clipping"""
        return torch.sigmoid(x - 2.0)

    def rot_act(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation normalization"""
        return F.normalize(x, dim=-1)

    def rgb_act(self, x: torch.Tensor) -> torch.Tensor:
        """RGB activation (always tanh for clip-based head)"""
        return 0.5 * torch.tanh(x) + 0.5

    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians using a single deconv layer.
        Supports both xyz and depth prediction modes.

        Args:
            x: Input tokens [B, N, C]
            rays_os: Ray origins [B, V, 3, H, W] (required for depth prediction)
            rays_ds: Ray directions [B, V, 3, H, W] (required for depth prediction)

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        B = x.shape[0]

        # Process through deconv
        x = x.permute(0, 2, 1).unsqueeze(-1)  # B, C, N, 1

        x = x.squeeze(-1).permute(0, 2, 1)  # B, N, C
        x = self.deconv(x)  # [B, N, output_dims * P * P]
        x = rearrange(x, "b n (p c) -> b c (n p)", p=self.opt.patch_size**2)

        x = x.reshape(B, self.output_dims, -1)  # B, output_dims, N * P * P
        x = x.permute(0, 2, 1).contiguous()  # B, N * P * P, output_dims

        # Apply activations based on prediction mode
        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)
        

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat(
            [pos, opacity, scale, rotation, rgbs], dim=-1
        )  # [B, N, 14]

        return gaussians
