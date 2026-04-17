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

"""PSNR, LPIPS, and SSIM for image evaluation (optional soft masks)."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from lpips import LPIPS
from skimage.metrics import structural_similarity


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of `x` weighted by optional soft `mask` in [0, 1] (broadcast to `x.shape`)."""
    eps = 1e-6
    if mask is None:
        return x.mean()
    mask = torch.broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(min=eps)


class MetricsCalculator:
    """Lazily constructs LPIPS on first use; supports 4D or 5D (B, V, C, H, W) batches."""

    def __init__(self, device="cuda", lpips_net="vgg"):
        self.device = device
        self.lpips_model = None
        self.lpips_net = lpips_net
        
    def _get_lpips(self):
        if self.lpips_model is None:
            self.lpips_model = LPIPS(net=self.lpips_net).to(self.device)
            self.lpips_model.requires_grad_(False)
            self.lpips_model.eval()
        return self.lpips_model
    
    @torch.no_grad()
    def calculate_psnr(self, pred_images, gt_images, mask=None, reduction="mean"):
        # Clip values to valid range [0, 1]
        pred_images = pred_images.clip(min=0, max=1)
        gt_images = gt_images.clip(min=0, max=1)
        
        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False
        
        # Compute MSE per image
        mse_per_pixel = (pred_images - gt_images) ** 2
        
        if mask is None:
            # No mask: compute mean over C, H, W dimensions
            mse = reduce(mse_per_pixel, "b c h w -> b", "mean")
        else:
            # With mask: compute masked mean per image
            psnr_list = []
            for i in range(mse_per_pixel.shape[0]):
                mse_val = masked_mean(mse_per_pixel[i], mask[i])
                psnr_list.append(mse_val)
            mse = torch.stack(psnr_list)
        
        psnr = -10 * mse.log10()
        
        # Reshape back if input was 5D
        if was_5d:
            psnr = psnr.reshape(B, V)
        
        if reduction == "mean":
            return psnr.mean()
        if reduction == "sum":
            return psnr.sum()
        return psnr

    @torch.no_grad()
    def calculate_lpips(self, pred_images, gt_images, mask=None, reduction="mean"):
        lpips_model = self._get_lpips()
        
        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False
        
        # Apply mask to images if provided
        if mask is not None:
            pred_images_masked = pred_images * mask
            gt_images_masked = gt_images * mask
        else:
            pred_images_masked = pred_images
            gt_images_masked = gt_images
        
        # Use normalize=True to handle [0, 1] -> [-1, 1] conversion automatically
        lpips_output = lpips_model.forward(gt_images_masked, pred_images_masked, normalize=True)
        
        if mask is None:
            # Extract the scalar value (LPIPS returns [B, 1, 1, 1])
            lpips_value = lpips_output[:, 0, 0, 0]
        else:
            # Compute masked mean over spatial dimensions
            lpips_value_list = []
            for i in range(lpips_output.shape[0]):
                # lpips_output is [B, 1, H', W'] where H', W' might be smaller than H, W
                lpips_map = lpips_output[i, 0]  # [H', W']
                
                # Resize mask to match LPIPS output spatial dimensions if needed
                if lpips_map.shape != mask[i, 0].shape:
                    mask_resized = F.interpolate(
                        mask[i : i + 1],
                        size=lpips_map.shape,
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                else:
                    mask_resized = mask[i, 0]
                
                lpips_val = masked_mean(lpips_map, mask_resized)
                lpips_value_list.append(lpips_val)
            lpips_value = torch.stack(lpips_value_list)
        
        # Reshape back if input was 5D
        if was_5d:
            lpips_value = lpips_value.reshape(B, V)
        
        if reduction == "mean":
            return lpips_value.mean()
        if reduction == "sum":
            return lpips_value.sum()
        return lpips_value

    @torch.no_grad()
    def calculate_ssim(self, pred_images, gt_images, mask=None, reduction="mean"):
        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False
        
        # Apply mask if provided
        if mask is not None:
            pred_images = pred_images * mask
            gt_images = gt_images * mask
        
        # Compute SSIM per image using skimage
        ssim_values = []
        for gt, pred in zip(gt_images, pred_images):
            ssim_val = structural_similarity(
                gt.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            ssim_values.append(ssim_val)
        
        ssim_tensor = torch.tensor(ssim_values, dtype=pred_images.dtype, device=pred_images.device)
        
        # Reshape back if input was 5D
        if was_5d:
            ssim_tensor = ssim_tensor.reshape(B, V)
        
        if reduction == "mean":
            return ssim_tensor.mean()
        if reduction == "sum":
            return ssim_tensor.sum()
        return ssim_tensor

    @torch.no_grad()
    def calculate_all_metrics(self, pred_images, gt_images, mask=None, reduction="mean"):
        return {
            "psnr": self.calculate_psnr(pred_images, gt_images, mask=mask, reduction=reduction),
            "lpips": self.calculate_lpips(pred_images, gt_images, mask=mask, reduction=reduction),
            "ssim": self.calculate_ssim(pred_images, gt_images, mask=mask, reduction=reduction),
        }


class MetricsTracker:
    """Append scalar metrics per step; `get_stats` / `print_summary` / `save_to_file` for reporting."""

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict=None, **kwargs):
        if metrics_dict is not None:
            if not isinstance(metrics_dict, dict):
                raise TypeError("metrics_dict must be a dictionary")
            kwargs.update(metrics_dict)
        
        for metric_name, value in kwargs.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[metric_name].append(value)
    
    def get_values(self, metric_name):
        if metric_name not in self.metrics:
            return np.array([])
        return np.array(self.metrics[metric_name])
    
    def get_stats(self, metric_name=None):
        if metric_name is not None:
            values = self.get_values(metric_name)
            if len(values) == 0:
                return None
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        stats = {}
        for name in self.metrics.keys():
            stats[name] = self.get_stats(name)
        return stats
    
    def get_mean(self, metric_name):
        stats = self.get_stats(metric_name)
        return stats["mean"] if stats else None

    def get_std(self, metric_name):
        stats = self.get_stats(metric_name)
        return stats["std"] if stats else None

    def reset(self):
        self.metrics = {}

    def _is_timing_metric(self, metric_name):
        timing_keywords = ("time", "duration", "latency", "elapsed")
        return any(keyword in metric_name.lower() for keyword in timing_keywords)

    def _format_metric_value(self, metric_name, value):
        if self._is_timing_metric(metric_name):
            if value < 1.0:
                return f"{value * 1000:.2f}ms"
            return f"{value:.3f}s"
        return f"{value:.4f}"
    
    def __len__(self):
        if not self.metrics:
            return 0
        return len(next(iter(self.metrics.values())))

    def __str__(self):
        stats = self.get_stats()
        lines = ["Metrics Statistics:"]
        for name, stat in stats.items():
            if stat:
                lines.append(
                    f"  {name.upper()}: {stat['mean']:.4f} ± {stat['std']:.4f} "
                    f"[{stat['min']:.4f}, {stat['max']:.4f}] (n={stat['count']})"
                )
        return "\n".join(lines)

    def print_summary(self, title="Evaluation Metrics", short=False):
        stats = self.get_stats()
        
        if short:
            metric_strs = []
            for name, stat in stats.items():
                if stat:
                    mean_str = self._format_metric_value(name, stat["mean"])
                    metric_strs.append(f"{name.upper()}: {mean_str}")
            print(f"[{title}] " + ", ".join(metric_strs))
        else:
            print("=" * 50)
            print(f"{title}:")
            for name, stat in stats.items():
                if stat:
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    print(f"  {name.upper():20s}: {mean_str:>10s} ± {std_str:<10s}")
            print("=" * 50)

    def save_to_file(self, filepath):
        stats = self.get_stats()
        
        with open(filepath, "w") as f:
            f.write("Evaluation Metrics:\n")
            f.write("=" * 70 + "\n")

            quality_metrics = {}
            timing_metrics = {}
            for name, stat in stats.items():
                if stat:
                    if self._is_timing_metric(name):
                        timing_metrics[name] = stat
                    else:
                        quality_metrics[name] = stat

            if quality_metrics:
                f.write("\nImage Quality Metrics:\n")
                f.write("-" * 70 + "\n")
                for name, stat in quality_metrics.items():
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    min_str = self._format_metric_value(name, stat["min"])
                    max_str = self._format_metric_value(name, stat["max"])
                    f.write(
                        f"{name.upper():20s}: {mean_str:>10s} ± {std_str:<10s} "
                        f"[{min_str:>10s}, {max_str:<10s}]\n"
                    )

            if timing_metrics:
                f.write("\nTiming Metrics:\n")
                f.write("-" * 70 + "\n")
                for name, stat in timing_metrics.items():
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    min_str = self._format_metric_value(name, stat["min"])
                    max_str = self._format_metric_value(name, stat["max"])
                    f.write(
                        f"{name.upper():20s}: {mean_str:>10s} ± {std_str:<10s} "
                        f"[{min_str:>10s}, {max_str:<10s}]\n"
                    )

            f.write("\n" + "=" * 70 + "\n")
            f.write("Per-sample metrics:\n")
            f.write("-" * 70 + "\n")

            n_samples = len(self)
            if n_samples > 0:
                metric_names = list(self.metrics.keys())
                f.write(f"{'Sample':<10}")
                for name in metric_names:
                    f.write(f"{name.upper():<15}")
                f.write("\n")
                f.write("-" * 70 + "\n")
                for i in range(n_samples):
                    f.write(f"{i:<10}")
                    for name in metric_names:
                        value = self.metrics[name][i]
                        value_str = self._format_metric_value(name, value)
                        f.write(f"{value_str:<15}")
                    f.write("\n")

        print(f"Metrics saved to {filepath}")

