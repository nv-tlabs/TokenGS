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

import numpy as np
import torch
import torchvision.transforms as transforms
from einops import einsum
from packaging import version as pver


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    """Sinusoidal embeddings for 1-D timesteps (fractional values allowed)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ImageTransform:
    """Center-crop, optional max-crop-to-fill, then resize; returns shift/scale for intrinsics."""

    def __init__(self, crop_size, sample_size, max_crop):
        self.crop_size = crop_size
        self.max_crop = max_crop
        self.sample_size = sample_size
        self.crop_transform = transforms.CenterCrop(crop_size) if crop_size else lambda x: x
        self.resize_transform = (
            transforms.Resize(sample_size) if sample_size else lambda x: x
        )

    def preprocess_images(self, images):
        video = images
        flip_flag = torch.zeros(images.shape[0], dtype=torch.bool, device=video.device)

        ori_h, ori_w = video.shape[-2:]
        if self.max_crop:
            crop_ratio = min(ori_h / self.crop_size[0], ori_w / self.crop_size[1])
            new_crop_size = (
                int(self.crop_size[0] * crop_ratio),
                int(self.crop_size[1] * crop_ratio),
            )
            self.crop_transform = transforms.CenterCrop(new_crop_size)

        video = self.crop_transform(video)
        new_h, new_w = video.shape[-2:]
        # u,v convention: shift relative to original (un-cropped) raster
        shift = ((new_w - ori_w) / 2, (new_h - ori_h) / 2)

        ori_h, ori_w = video.shape[-2:]
        video = self.resize_transform(video)
        new_h, new_w = video.shape[-2:]
        scale = (new_w / ori_w, new_h / ori_h)

        return video, shift, scale, flip_flag

    def apply_img_transform(self, i, j, shift, scale):
        """Map pixel coordinates from pre-crop space to post-resize space (shift then scale)."""
        i = (i + shift[0]) * scale[0]
        j = (j + shift[1]) * scale[1]
        return i, j



def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing="ij")


def get_grid_uvs(batch_shape, H, W, device, dtype=None, flip_flag=None, nh=None, nw=None, margin=0):
    if dtype is None:
        dtype = torch.float32
    if nh is None:
        nh = H
    if nw is None:
        nw = W
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # c2w @ dirctions
    B, V = batch_shape

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
        torch.linspace(0, W - 1, nw, device=device, dtype=dtype),
    )
    i = i.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5
    j = j.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5

    if margin != 0:
        marginw = 1 - 2 * margin
        i = marginw * i + margin * W
        j = marginw * j + margin * H

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
            torch.linspace(W - 1, 0, nw, device=device, dtype=dtype),
        )
        i_flip = i_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        j_flip = j_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip
    return i, j


def get_rays_from_uvs(i, j, K, c2w):
    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)
    return rays_o, rays_d


def get_rays(K, c2w, H, W, device, flip_flag=None, nh=None, nw=None):
    i, j = get_grid_uvs(
        K.shape[:2], H=H, W=W, dtype=K.dtype, device=device, flip_flag=flip_flag, nh=nh, nw=nw
    )
    return get_rays_from_uvs(i, j, K, c2w)


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    B, _ = K.shape[:2]
    rays_o, rays_d = get_rays(K, c2w, H, W, device, flip_flag=flip_flag)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)[0].permute(0, 3, 1, 2).contiguous()
    rays_o = rays_o.reshape(B, c2w.shape[1], H, W, 3)[0].permute(0, 3, 1, 2).contiguous()
    rays_d = rays_d.reshape(B, c2w.shape[1], H, W, 3)[0].permute(0, 3, 1, 2).contiguous()
    return plucker, rays_o, rays_d


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)