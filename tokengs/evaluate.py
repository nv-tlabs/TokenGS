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

"""Run a checkpoint on the test loader; supports optional `--config` YAML for tyro defaults."""

import os
import sys
import time
import warnings
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn as nn
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm

from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.options import AllConfigs, Options
from tokengs.utils import MetricsCalculator, MetricsTracker
from tokengs.utils.images import save_image_grid_square, visualize_depth

warnings.filterwarnings("ignore")

torch.manual_seed(0)


def _pop_config_path_from_argv() -> str | None:
    """If argv contains `--config PATH`, remove both tokens and return PATH."""
    if "--config" not in sys.argv:
        return None
    idx = sys.argv.index("--config")
    if idx + 1 >= len(sys.argv):
        return None
    path = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return path


def _load_options(accelerator: Accelerator, config_path: str | None) -> Options:
    if config_path is not None:
        model_path = os.path.join(os.path.dirname(config_path), "model.safetensors")
        accelerator.print(f"[INFO] loading config from --config={config_path}")
        with open(config_path, encoding="utf-8") as f:
            default_opt = tyro.extras.from_yaml(Options, f)
        opt = tyro.cli(Options, default=default_opt)
        if opt.resume is None:
            accelerator.print(f"[INFO] resume not provided, deducing {model_path=} from {config_path=}")
            if not os.path.exists(model_path):
                raise ValueError(f"{model_path=} does not exist")
            opt.resume = model_path
        return opt

    accelerator.print("[INFO] no config provided, using default options")
    return tyro.cli(AllConfigs)


def _load_checkpoint_state_dict(resume: str) -> dict[str, Any]:
    if resume.endswith("safetensors"):
        return load_file(resume, device="cpu")
    return torch.load(resume, map_location="cpu", weights_only=False)


def _copy_matching_checkpoint(model: nn.Module, ckpt: dict[str, Any], log) -> None:
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k not in state_dict:
            log(f"[WARN] unexpected param {k}: {v.shape}")
            continue
        if state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
        else:
            log(
                f"[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored."
            )


def _copy_checkpoint_strict(model: nn.Module, ckpt: dict[str, Any]) -> None:
    state_dict = model.state_dict()
    unexpected = []
    mismatched = []
    loaded = set()

    for k, v in ckpt.items():
        if "lpips_loss" in k:
            continue
        if k not in state_dict:
            unexpected.append(k)
            continue
        if state_dict[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(state_dict[k].shape)))
            continue
        state_dict[k].copy_(v)
        loaded.add(k)

    missing = [
        k for k in state_dict.keys()
        if "lpips_loss" not in k and k not in loaded
    ]
    if unexpected or mismatched or missing:
        msg = ["Checkpoint does not match model preset."]
        if unexpected:
            msg.append(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
        if mismatched:
            msg.append(f"Mismatched shapes ({len(mismatched)}): {mismatched[:20]}")
        if missing:
            msg.append(f"Missing keys ({len(missing)}): {missing[:20]}")
        raise RuntimeError("\n".join(msg))


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _views_chw_to_uint8_nhwc(views: torch.Tensor) -> np.ndarray:
    """(V, C, H, W) float in ~[0, 1] -> (V, H, W, C) uint8."""
    x = views.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def _blend_mask_overlay(vis: torch.Tensor, masks: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return vis * masks + alpha * (1 - masks) * vis + (1 - alpha) * (1 - masks)


def main() -> None:
    accelerator = Accelerator()
    config_path = _pop_config_path_from_argv()
    opt = _load_options(accelerator, config_path)
    opt.use_input_supervision = False

    ttt_status = f"using {opt.ttt_mode} TTT" if opt.use_ttt_for_eval else "not using TTT"
    accelerator.print(f"[INFO] {ttt_status} for evaluation")

    model = model_registry[opt.model_type](opt)
    if opt.resume is None or opt.resume == "None":
        raise ValueError("Resume path is required when evaluating a model")

    accelerator.print(f"[INFO] loading model from {opt.resume=}")
    ckpt = _load_checkpoint_state_dict(opt.resume)
    if opt.strict_checkpoint_loading:
        _copy_checkpoint_strict(model, ckpt)
    else:
        _copy_matching_checkpoint(model, ckpt, accelerator.print)
    accelerator.print("[INFO] Model loaded!")

    _train_dl, test_dataloader, _, _ = get_multi_dataloader(opt, accelerator)
    model, _train_dl, test_dataloader = accelerator.prepare(model, _train_dl, test_dataloader)
    model.eval()
    base_model = accelerator.unwrap_model(model)

    ws = opt.workspace
    os.makedirs(ws, exist_ok=True)
    if opt.eval_n_media_dumps > 0:
        for sub in ("images", "videos", "gaussians", "depths"):
            os.makedirs(os.path.join(ws, sub), exist_ok=True)

    metrics_calc = MetricsCalculator(device=accelerator.device)
    metrics_tracker = MetricsTracker()
    worker_rank = accelerator.process_index

    pbar = tqdm(test_dataloader, disable=not accelerator.is_main_process)
    with torch.no_grad(), pbar:
        for i, data in enumerate(pbar):
            _cuda_sync()
            t0 = time.perf_counter()
            results = model(data)
            _cuda_sync()
            inference_time = time.perf_counter() - t0

            input_images = data["images_input"]
            pred_images = results["images_pred"]
            gt_images = data["images_output"]
            gaussians = results["gaussians"]
            masks = data["masks_output"]
            has_mask = data["has_mask"]
            depth_maps = results.get("depths_pred")

            mask_kw = {"mask": masks} if has_mask.any() else {}
            metrics = metrics_calc.calculate_all_metrics(
                pred_images, gt_images, reduction="mean", **mask_kw
            )
            metrics["inference_time"] = inference_time

            gathered = accelerator.gather(
                {k: torch.tensor(v, device=accelerator.device) for k, v in metrics.items()}
            )
            metrics_tracker.update({k: v.mean().item() for k, v in gathered.items()})

            postfix: dict[str, str] = {}
            if torch.cuda.is_available():
                mem_free, mem_total = torch.cuda.mem_get_info()
                postfix["mem_used"] = f"{(mem_total - mem_free) / 1024**3:.1f}GB"
            for k, st in metrics_tracker.get_stats().items():
                if st:
                    postfix[k] = f"{st['mean']:.4f} ± {st['std']:.4f}"
            pbar.set_postfix(**postfix)

            if i < opt.eval_n_media_dumps:
                pred_vis = pred_images.detach().clone()
                gt_vis = gt_images.detach().clone()
                input_vis = input_images.detach().clone()
                if has_mask.any():
                    pred_vis = _blend_mask_overlay(pred_vis, masks)
                    gt_vis = _blend_mask_overlay(gt_vis, masks)

                for b_idx in range(pred_images.shape[0]):
                    tag = f"{b_idx}-{worker_rank}-{i}"
                    save_image_grid_square(
                        input_vis[b_idx], os.path.join(ws, "images", f"input_{tag}.png")
                    )
                    save_image_grid_square(pred_vis[b_idx], os.path.join(ws, "images", f"pred_{tag}.png"))
                    save_image_grid_square(gt_vis[b_idx], os.path.join(ws, "images", f"gt_{tag}.png"))

                    if depth_maps is not None:
                        depth_vis = visualize_depth(depth_maps[b_idx])
                        save_image_grid_square(
                            depth_vis, os.path.join(ws, "depths", f"depth_{tag}.png")
                        )
                        depth_u8 = _views_chw_to_uint8_nhwc(depth_vis)
                        imageio.mimwrite(
                            os.path.join(ws, "videos", f"depth_{tag}.mp4"), depth_u8, fps=30
                        )

                    pred_u8 = _views_chw_to_uint8_nhwc(pred_vis[b_idx])
                    input_u8 = _views_chw_to_uint8_nhwc(input_vis[b_idx])
                    gt_u8 = _views_chw_to_uint8_nhwc(gt_vis[b_idx])
                    combined = np.concatenate([gt_u8, pred_u8], axis=-2)

                    for name, arr in (
                        ("comparison", combined),
                        ("pred", pred_u8),
                        ("gt", gt_u8),
                        ("input", input_u8),
                    ):
                        imageio.mimwrite(os.path.join(ws, "videos", f"{name}_{tag}.mp4"), arr, fps=30)

                    base_model.gs.save_ply(
                        gaussians[b_idx : b_idx + 1],
                        os.path.join(ws, "gaussians", f"gaussians_{tag}.ply"),
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        metrics_tracker.print_summary(title="Final Evaluation Metrics")
        metrics_tracker.save_to_file(os.path.join(ws, "metrics.txt"))

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
