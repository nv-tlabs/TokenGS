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

import imageio
import matplotlib
import numpy as np
import torch
from einops import rearrange


def _turbo_colormap():
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps["turbo"]
    import matplotlib.cm as cm

    return cm.get_cmap("turbo")


def image_views_to_square_grid(images: torch.Tensor) -> torch.Tensor:
    """Arrange views (v, c, h, w) in a roughly square grid (H, W, c)."""
    assert images.ndim == 4, f"Expected 4D tensor (v, c, h, w), got {images.ndim}D"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"

    n_views = images.shape[0]
    n_cols = math.ceil(math.sqrt(n_views))
    n_rows = math.ceil(n_views / n_cols)

    n_needed = n_rows * n_cols
    if n_needed > n_views:
        padding = torch.zeros(
            n_needed - n_views,
            images.shape[1],
            images.shape[2],
            images.shape[3],
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat([images, padding], dim=0)

    images = images.reshape(n_rows, n_cols, images.shape[1], images.shape[2], images.shape[3])
    grid = rearrange(images, "r c ch h w -> (r h) (c w) ch")
    
    return grid


def save_image_grid_square(images: torch.Tensor, output_path: str):
    """Save (v, c, h, w) tensor as one PNG grid via `image_views_to_square_grid`."""
    images = (images.clamp(0, 1) * 255).to(torch.uint8)
    images_grid = image_views_to_square_grid(images)
    imageio.imwrite(output_path, images_grid.cpu().numpy())


def visualize_depth(depth: torch.Tensor, vmin: float | None = None, vmax: float | None = None) -> torch.Tensor:
    """Map depth (v, 1, h, w) or (v, h, w) to RGB (v, 3, h, w) in [0, 1] using the turbo colormap."""
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    assert depth.ndim == 3, f"Expected 3D tensor (v, h, w), got {depth.ndim}D"

    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()

    depth_normalized = ((depth - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    depth_np = depth_normalized.cpu().numpy()
    cmap = _turbo_colormap()
    rgba = cmap(depth_np.reshape(-1)).reshape(*depth_np.shape, 4)
    colored = torch.from_numpy(rgba[..., :3]).float().permute(0, 3, 1, 2)
    return colored.to(depth.device)

