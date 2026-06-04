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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tokengs.options import Options


IndexSpec = int | slice


def _preserve_dim(index: IndexSpec) -> slice:
    if isinstance(index, int):
        return slice(index, index + 1)
    return index


@dataclass
class ModelInputEncoder:
    """Input data for the encoder (input views)"""
    # Image features split by type (input views only)
    images_rgb: torch.Tensor  # [B, V_in, 3, H, W] - RGB channels (normalized for encoder)
    plucker: torch.Tensor  # [B, V_in, 6, H, W] - Plucker coordinates

    # Ray information (input views only)
    rays_os: torch.Tensor  # [B, V_in, 3, H, W] - ray origins
    rays_ds: torch.Tensor  # [B, V_in, 3, H, W] - ray directions

    # Camera parameters for input views
    intrinsics_input: torch.Tensor  # [B, V_in, 4]
    cam_to_world_input: torch.Tensor  # [B, V_in, 4, 4]

    # Optional time embeddings for input views
    time_embedding_input: Optional[torch.Tensor] = None  # [B, V_in, T_dim, H, W]
    time_embedding_target: Optional[torch.Tensor] = None  # [B, V_out, T_dim, H, W]

    # Unnormalized RGB images for supervision (e.g., TTT)
    images_rgb_unnormalized: Optional[torch.Tensor] = None  # [B, V_in, 3, H, W]


@dataclass
class EncoderLatent:
    """Encoder output containing keys and values for cross-attention"""
    keys: torch.Tensor  # [B, H, N, C//H] - attention keys
    values: torch.Tensor  # [B, H, N, C//H] - attention values


@dataclass
class ModelInputDecoder:
    """Input data for the decoder (target views)"""
    # Target time embedding (used to condition GS tokens)
    time_embedding_target: Optional[torch.Tensor] = None  # [B, 1, T_dim, H, W]

    # Camera parameters for rendering (output views)
    cam_view: Optional[torch.Tensor] = None  # [B, V_out, 4, 4] - Camera view matrices for rendering
    intrinsics: Optional[torch.Tensor] = None  # [B, V_out, 4] - Intrinsics for rendering

    def select_batch(self, scenes: IndexSpec, views: Optional[IndexSpec] = None) -> ModelInputDecoder:
        scene_slice = _preserve_dim(scenes)
        view_slice = _preserve_dim(views) if views is not None else slice(None)
        return type(self)(
            time_embedding_target=(
                None if self.time_embedding_target is None else self.time_embedding_target[scene_slice]
            ),
            cam_view=None if self.cam_view is None else self.cam_view[scene_slice, view_slice],
            intrinsics=None if self.intrinsics is None else self.intrinsics[scene_slice, view_slice],
        )


@dataclass
class Reconstruction:
    """Model prediction ready for rendering."""
    gaussians: torch.Tensor  # [B, N, 14] - activated Gaussian parameters
    background_color: torch.Tensor  # [3] - RGB background color


@dataclass
class ModelInput:
    """Complete input data for the TokenGS model (combines encoder and decoder inputs)"""
    encoder: ModelInputEncoder
    decoder: ModelInputDecoder

    def to_ttt(self, masks_output: Optional[torch.Tensor] = None, has_mask: bool = False) -> tuple[ModelInput, ModelSupervision]:
        """
        Create ModelInput and ModelSupervision for test-time training.

        For TTT, we want to:
        1. Render back to the input views (not target views)
        2. Supervise using the input view RGB images as ground truth

        Args:
            masks_output: Optional masks [B, V_in, 1, H, W]. If None, creates all-ones masks.
            has_mask: Whether masks are present/should be used

        Returns:
            Tuple of (ModelInput for TTT, ModelSupervision for TTT)
        """
        cam_view_input = torch.inverse(self.encoder.cam_to_world_input).transpose(-2, -1)

        decoder_input_ttt = ModelInputDecoder(
            time_embedding_target=None,
            cam_view=cam_view_input,
            intrinsics=self.encoder.intrinsics_input,
        )

        model_input_ttt = ModelInput(
            encoder=self.encoder,
            decoder=decoder_input_ttt,
        )

        assert self.encoder.images_rgb_unnormalized is not None, (
            "TTT requires unnormalized images for supervision. "
            "Ensure split_data() is called with a batch containing 'images_input'."
        )
        images_for_supervision = self.encoder.images_rgb_unnormalized

        if masks_output is None:
            B, V_in, _, H, W = images_for_supervision.shape
            masks_output = torch.ones(
                B, V_in, 1, H, W,
                dtype=images_for_supervision.dtype,
                device=images_for_supervision.device
            )
        else:
            B = images_for_supervision.shape[0]

        has_mask_tensor = torch.full(
            (B,),
            has_mask,
            dtype=torch.bool,
            device=images_for_supervision.device
        )

        supervision_ttt = ModelSupervision(
            images_output=images_for_supervision,
            masks_output=masks_output,
            has_mask=has_mask_tensor,
            rays_os=self.encoder.rays_os,
            rays_ds=self.encoder.rays_ds,
        )

        return model_input_ttt, supervision_ttt

    @property
    def batch_size(self) -> int:
        return self.encoder.images_rgb.shape[0]


@dataclass
class ModelSupervision:
    """Supervision data for the TokenGS model"""
    images_output: torch.Tensor  # [B, V, 3, img_size, img_size] - ground truth images
    masks_output: torch.Tensor  # [B, V, 1, img_size, img_size] - ground truth masks
    has_mask: torch.Tensor  # [B] - boolean mask indicating if a given scene has masks

    # Ray information for 3D visibility loss
    rays_os: Optional[torch.Tensor] = None  # [B, V, 3, H, W] - ray origins
    rays_ds: Optional[torch.Tensor] = None  # [B, V, 3, H, W] - ray directions

    def __post_init__(self):
        assert (self.rays_os is not None) == (self.rays_ds is not None), "rays_os and rays_ds must be either both None or both not None"

    def select_batch(self, scenes: IndexSpec, views: Optional[IndexSpec] = None) -> ModelSupervision:
        scene_slice = _preserve_dim(scenes)
        view_slice = _preserve_dim(views) if views is not None else slice(None)
        return type(self)(
            images_output=self.images_output[scene_slice, view_slice],
            masks_output=self.masks_output[scene_slice, view_slice],
            has_mask=self.has_mask[scene_slice],
            rays_os=None if self.rays_os is None else self.rays_os[scene_slice, view_slice],
            rays_ds=None if self.rays_ds is None else self.rays_ds[scene_slice, view_slice],
        )


def split_data(batch: dict, opt: Options) -> tuple[ModelInput, ModelSupervision]:
    """
    Split the batch dictionary into structured ModelInput and ModelSupervision objects.

    Args:
        batch: Dictionary from the dataloader containing mixed input and supervision data
        opt: Options object containing model configuration

    Returns:
        Tuple of (ModelInput, ModelSupervision)
    """
    images = batch['input']  # C = 3 (RGB) + time_embedding_dim + 6 (plucker)

    images_rgb_all = images[:, :, :3, :, :]
    plucker_all = images[:, :, -6:, :, :]

    time_embedding_input = None
    time_embedding_target = None
    if opt.time_embedding:
        time_embedding_input = images[:, :opt.num_input_views, 3:3+opt.time_embedding_dim, :, :]
        time_embedding_target = images[:, opt.num_input_views:opt.num_input_views+1, 3:3+opt.time_embedding_dim, :, :]

    if opt.use_input_supervision:
        cam_view = batch['cam_view_all']
        intrinsics = batch['intrinsics_all']
        images_output = batch['images_all']
        masks_output = batch['masks_all']
    else:
        cam_view = batch['cam_view']
        intrinsics = batch['intrinsics']
        images_output = batch['images_output']
        masks_output = batch['masks_output']

    encoder_input = ModelInputEncoder(
        images_rgb=images_rgb_all[:, :opt.num_input_views],
        plucker=plucker_all[:, :opt.num_input_views],
        rays_os=batch['rays_os'][:, :opt.num_input_views],
        rays_ds=batch['rays_ds'][:, :opt.num_input_views],
        intrinsics_input=batch['intrinsics_input'],
        cam_to_world_input=batch['cam_to_world_input'],
        time_embedding_input=time_embedding_input,
        time_embedding_target=time_embedding_target,
        images_rgb_unnormalized=batch['images_input'],
    )

    decoder_input = ModelInputDecoder(
        time_embedding_target=time_embedding_target,
        cam_view=cam_view,
        intrinsics=intrinsics,
    )

    model_input = ModelInput(
        encoder=encoder_input,
        decoder=decoder_input,
    )

    supervision = ModelSupervision(
        images_output=images_output,
        masks_output=masks_output,
        has_mask=batch['has_mask'],
        rays_os=batch['rays_os'],
        rays_ds=batch['rays_ds'],
    )

    return model_input, supervision
