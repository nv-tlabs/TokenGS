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

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.rendering.fused_ssim import fused_ssim
from tokengs.models.input_types import ModelInputDecoder, ModelSupervision
from tokengs.options import Options


def project_gaussian_means2d(
    gaussians_xyz: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project Gaussian centers with the same camera convention used by the renderer."""
    gaussians_xyz = gaussians_xyz.float()
    viewmats = cam_view.to(device=gaussians_xyz.device, dtype=gaussians_xyz.dtype).transpose(-1, -2)
    intrinsics = intrinsics.to(device=gaussians_xyz.device, dtype=gaussians_xyz.dtype)

    rotation = viewmats[..., :3, :3]
    translation = viewmats[..., :3, 3]
    means_camera = torch.einsum("bvij,bnj->bvni", rotation, gaussians_xyz) + translation[:, :, None, :]

    x, y, z = means_camera.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    u = fx[:, :, None] * x / z + cx[:, :, None]
    v = fy[:, :, None] * y / z + cy[:, :, None]
    return torch.stack((u, v), dim=-1)


def compute_visibility_loss_from_means2d(
    opt: Options,
    img_size: tuple[int, int] | list[int],
    means2d_pred: torch.Tensor,
) -> torch.Tensor:
    """Per-scene visibility loss from projected Gaussian centers."""
    H, W = int(img_size[0]), int(img_size[1])
    uv = torch.nan_to_num(means2d_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    scale = torch.tensor([W, H], dtype=uv.dtype, device=uv.device)
    uv_norm = (uv / scale) * 2 - 1
    out_of_bounds = F.relu(torch.abs(uv_norm) - 1.0)
    out_of_bounds = out_of_bounds.sum(-1)
    vis_loss = out_of_bounds.min(dim=1).values
    if opt.visibility_distance_threshold > 0:
        vis_loss = vis_loss.clamp(max=opt.visibility_distance_threshold)
    return vis_loss.mean(dim=-1)


def compute_loss_from_renders(
    opt: Options,
    img_size: tuple[int, int] | list[int],
    pred_images: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_images: torch.Tensor,
    gt_masks: torch.Tensor,
    has_mask: torch.Tensor,
    lpips_loss: Optional[nn.Module] = None,
) -> dict:
    """Per-scene loss from rendered predictions (intended for use inside torch.vmap)."""
    H, W = int(img_size[0]), int(img_size[1])
    results: dict = {}
    assert has_mask.shape == (pred_images.shape[0],), "has_mask must have the same batch size as pred_images"

    bg_color = torch.ones(3, dtype=pred_images.dtype, device=pred_images.device)
    gt_images_lpips = gt_images
    gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

    if opt.rgb_loss_type == "l1":
        loss_rgb = F.l1_loss(pred_images, gt_images)
    else:
        loss_rgb = F.mse_loss(pred_images, gt_images)
    loss = opt.lambda_rgb * loss_rgb
    results["loss_rgb"] = loss_rgb

    if opt.lambda_mask > 0:
        loss_mse_mask = F.mse_loss(pred_masks, gt_masks, reduction="none").mean(dim=(1, 2, 3, 4))
        loss_mse_mask = torch.where(has_mask, loss_mse_mask, torch.zeros_like(loss_mse_mask)).sum()
        results["loss_mask"] = loss_mse_mask
        loss = loss + opt.lambda_mask * loss_mse_mask

    if opt.lambda_ssim > 0:
        ssim = fused_ssim(
            pred_images.view(-1, 3, H, W),
            gt_images.view(-1, 3, H, W),
        )
        loss_ssim = (1 - ssim) / 2
        loss = loss + opt.lambda_ssim * loss_ssim
        results["loss_ssim"] = loss_ssim

    lambda_lpips = float(opt.lambda_lpips)
    if lambda_lpips > 0 and lpips_loss is not None:
        loss_lpips = lpips_loss(
            gt_images_lpips.view(-1, 3, H, W),
            pred_images.view(-1, 3, H, W),
            normalize=True,
        ).mean()
        results["loss_lpips"] = loss_lpips

    results["loss"] = loss

    with torch.no_grad():
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2, dim=(-1, -2, -3)))
        results["psnr"] = psnr.mean()

    return results


def compute_tokengs_loss(
    opt: Options,
    img_size: tuple[int, int] | list[int],
    render_results: dict,
    supervision: ModelSupervision,
    decoder_input: ModelInputDecoder,
    gaussians: torch.Tensor,
    lpips_loss: Optional[nn.Module],
) -> dict:
    """Aggregate training loss from renderer outputs and supervision."""
    pred_images = render_results["images_pred"]
    pred_masks = render_results["alphas_pred"]
    depths_pred = render_results["depths_pred"]
    gt_images = supervision.images_output

    gaussians_xyz = gaussians[..., :3]
    per_scene = partial(compute_loss_from_renders, opt, img_size, lpips_loss=lpips_loss)

    results_per_scene = torch.vmap(per_scene, in_dims=(0,) * 5)(
        pred_images.unsqueeze(1),
        pred_masks.unsqueeze(1),
        supervision.images_output.unsqueeze(1),
        supervision.masks_output.unsqueeze(1),
        supervision.has_mask.unsqueeze(1),
    )

    if "loss_mask" in results_per_scene:
        results_per_scene["loss_mask"] = results_per_scene["loss_mask"] / (supervision.has_mask.float().sum() + 1e-6)

    if "loss_lpips" in results_per_scene:
        results_per_scene["loss"] = results_per_scene["loss"] + float(opt.lambda_lpips) * results_per_scene["loss_lpips"]

    if opt.lambda_visibility > 0:
        if decoder_input.cam_view is None:
            raise ValueError("decoder_input.cam_view is required for visibility loss")
        if decoder_input.intrinsics is None:
            raise ValueError("decoder_input.intrinsics is required for visibility loss")
        means2d_pred = project_gaussian_means2d(gaussians_xyz, decoder_input.cam_view, decoder_input.intrinsics)
        loss_visibility = compute_visibility_loss_from_means2d(opt, img_size, means2d_pred)
        results_per_scene["loss_visibility"] = loss_visibility
        results_per_scene["loss"] = results_per_scene["loss"] + opt.lambda_visibility * loss_visibility

    result_means = {k: v.mean() for k, v in results_per_scene.items()}
    results_per_scene = {f"{k}_per_scene": v for k, v in results_per_scene.items()}

    if opt.lambda_opacity > 0:
        opacity = gaussians[..., 3:4]
        loss_opacity = -torch.log(opacity + 1e-6).mean()
        result_means["loss_opacity"] = loss_opacity
        result_means["loss"] = result_means["loss"] + opt.lambda_opacity * loss_opacity

    return {
        **result_means,
        **results_per_scene,
        "images_pred": pred_images,
        "alphas_pred": pred_masks,
        "images_output": gt_images,
        "depths_pred": depths_pred,
    }
