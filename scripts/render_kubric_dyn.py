# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Render Kubric dynamic checkpoint videos, including input-camera hstacks.

Videos produced in --out_dir:
  01_gt_input.mp4         GT 24-frame video from one kubric camera (no inference)
  02_gt_trajectory.mp4    Render along the GT 24-frame trajectory of the same
                           camera, with the target time varying frame-by-frame.
  03_fixed_cam.mp4        Render from the 0th-frame camera (fixed pose), with
                           the target time varying frame-by-frame.
  04_fixed_cam_dyn_only.mp4 Same as #3 but rendering only the gaussians produced
                           from dynamic GS tokens (last num_dynamic_gs_tokens).
  05_fixed_cam_static_only.mp4 Same as #3 but rendering only static-token gaussians.
  06_input_cams_hstack.mp4     Render each input camera over time and hstack views.
  07_input_cams_hstack_dyn.mp4 Dynamic-token-only input-camera hstack.
  08_input_cams_hstack_static.mp4 Static-token-only input-camera hstack.

The model encoder is run once (4 input views at fixed (cam,time) pairs).
For each output frame we re-run the decoder with the desired target time and
swap in the desired target camera at render time.
"""
import argparse
import os
import sys

# Ensure the repo root is importable when running the script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision import transforms

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_IMAGE_RGB,
)
from tokengs.data.dynamic.kubric import Kubric
from tokengs.models import model_registry
from tokengs.models.input_types import (
    ModelInputDecoder,
    ModelInputEncoder,
)
from tokengs.options import Options, config_defaults
from tokengs.utils.data import ImageTransform, ray_condition, timestep_embedding


def load_options(workspace: str | None, preset: str) -> Options:
    import tyro
    if workspace is not None:
        p = os.path.join(workspace, "config.yaml")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return tyro.extras.from_yaml(Options, f)
    if preset not in config_defaults:
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {sorted(config_defaults)}")
    return config_defaults[preset].evolve()


def normalize_mean_cam(c2ws: torch.Tensor, num_input: int) -> torch.Tensor:
    """Replicate Provider._normalize_camera_mean_cam: normalize all c2ws by
    the mean frame of the first ``num_input`` views."""
    input_c2ws = c2ws[:num_input]
    position_avg = input_c2ws[:, :3, 3].mean(0)
    forward_avg = input_c2ws[:, :3, 2].mean(0)
    down_avg = input_c2ws[:, :3, 1].mean(0)
    forward_avg = F.normalize(forward_avg, dim=0)
    down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
    right_avg = torch.cross(down_avg, forward_avg, dim=0)
    pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)
    pos_avg = torch.cat(
        [pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0
    )
    pos_avg_inv = torch.inverse(pos_avg)
    return torch.matmul(pos_avg_inv.unsqueeze(0), c2ws)


def compute_pointmap_scale(
    raw_depths: torch.Tensor,
    c2ws: torch.Tensor,
    raw_intrinsics: torch.Tensor,
    num_input: int,
    trim_lo: float = 0.0,
    trim_hi: float = 1.0,
) -> float:
    """Replicate Provider._compute_pointmap_scale on the input depths only."""
    V = min(num_input, raw_depths.shape[0])
    all_dists = []
    for i in range(V):
        depth = raw_depths[i, 0]
        valid = depth > 1e-8
        if not valid.any():
            continue
        fx, fy, cx, cy = raw_intrinsics[i].tolist()
        H, W = depth.shape
        u = torch.arange(W, dtype=torch.float32)
        v = torch.arange(H, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        x_cam = (uu - cx) / fx * depth
        y_cam = (vv - cy) / fy * depth
        z_cam = depth
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        R = c2ws[i, :3, :3]
        t = c2ws[i, :3, 3]
        pts_world = pts_cam[valid] @ R.T + t
        all_dists.append(pts_world.norm(dim=-1))
    if not all_dists:
        return 1.0
    all_dists = torch.cat(all_dists, dim=0)
    if not (0.0 <= trim_lo < trim_hi <= 1.0):
        trim_lo, trim_hi = 0.0, 1.0
    if (trim_lo > 0.0 or trim_hi < 1.0) and all_dists.numel() > 1:
        q_lo = torch.quantile(all_dists, trim_lo) if trim_lo > 0.0 else all_dists.min()
        q_hi = torch.quantile(all_dists, trim_hi) if trim_hi < 1.0 else all_dists.max()
        trimmed = all_dists[(all_dists >= q_lo) & (all_dists <= q_hi)]
        if trimmed.numel() > 0:
            all_dists = trimmed
    return all_dists.mean().clamp(min=1e-6).item()


def write_mp4(frames_uint8: np.ndarray, path: str, fps: int = 8) -> None:
    """frames_uint8: (T, H, W, 3) uint8 array."""
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as w:
        for f in frames_uint8:
            w.append_data(f)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None,
                    help="Optional workspace containing config.yaml; falls back to --preset if absent.")
    ap.add_argument("--preset", default="finetune_dl3dv_kubric_dyn_release",
                    help="Options preset used when --workspace/config.yaml is unavailable.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument("--n_scenes", type=int, default=1,
                    help="Render this many consecutive scenes starting at scene_idx.")
    ap.add_argument("--gt_cam_index", type=int, default=0,
                    help="Which kubric camera to use for GT trajectory + fixed cam")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--kubric_root", default="data/kubric")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Load options + build model -----
    opt: Options = load_options(args.workspace, args.preset).evolve(
        evaluating=True,
        batch_size=1,
        use_input_supervision=False,
    )

    ckpt = args.ckpt
    if ckpt is None and args.workspace is not None:
        ckpt = os.path.join(args.workspace, "model.safetensors")
    if ckpt is None:
        raise ValueError("--ckpt is required when --workspace is omitted")
    print(f"[render] workspace={args.workspace}")
    print(f"[render] preset={args.preset}")
    print(f"[render] ckpt={ckpt}")
    print(f"[render] scene_idx={args.scene_idx} gt_cam_index={args.gt_cam_index}")
    print(f"[render] num_dynamic_gs_tokens={opt.num_dynamic_gs_tokens}")
    print(f"[render] num_gs_tokens={opt.num_gs_tokens}")

    model_cls = model_registry[opt.model_type]
    model = model_cls(opt).to(device).eval()
    state = load_file(ckpt, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # We expect LPIPS to be absent; ignore.
    missing = [k for k in missing if "lpips_loss" not in k]
    unexpected = [k for k in unexpected if "lpips_loss" not in k]
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  e.g. missing[0]={missing[0]}")

    # ----- Pull data from kubric for our chosen scene -----
    ds = Kubric(args.kubric_root, load_depth=(opt.camera_scale_method == "pointmap"))
    print(f"[ds] {len(ds)} scenes")
    base_out_dir = args.out_dir
    scene_indices = range(args.scene_idx, args.scene_idx + args.n_scenes)
    for idx in scene_indices:
        scene_out_dir = (base_out_dir if args.n_scenes == 1
                         else os.path.join(base_out_dir, f"scene_{idx:03d}"))
        os.makedirs(scene_out_dir, exist_ok=True)
        print(f"[scene] idx={idx} -> {scene_out_dir}")
        F_total = args.num_frames
        cam_gt = args.gt_cam_index
        V_in = opt.num_input_views  # 4

        # Choose 4 input (camera, frame) pairs.  Match the dynamic provider's
        # diagonal-grid layout: camera v at frame f_v, where f_v is evenly spaced
        # across [0..F-1].
        input_cam_indices = [(cam_gt + v + 1) % ds._num_cameras for v in range(V_in)]
        # avoid using cam_gt itself as one of the inputs (so video 3 uses an unseen-pose camera)
        input_frame_indices = [int(round(v * (F_total - 1) / (V_in - 1))) for v in range(V_in)]
        print(f"[inputs] cams={input_cam_indices} frames={input_frame_indices}")

        fields = [DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS]
        if opt.camera_scale_method == "pointmap":
            fields.append(DF_DEPTH)

        # 1. Load 4 input views (one frame per camera, view-major iteration).
        input_pack = []
        for cam_idx, frame_idx in zip(input_cam_indices, input_frame_indices):
            one = ds.get_data(
                idx,
                data_fields=fields,
                frame_indices=[frame_idx],
                view_indices=[cam_idx],
                num_depth_frames=1,
            )
            input_pack.append(one)

        def stack(field):
            return torch.cat([p[field] for p in input_pack], dim=0)

        rgbs_in = stack(DF_IMAGE_RGB)            # (V_in, 3, H_native, W_native)
        c2ws_in = stack(DF_CAMERA_C2W_TRANSFORM) # (V_in, 4, 4)
        intr_in = stack(DF_CAMERA_INTRINSICS)    # (V_in, 4)
        depth_in = stack(DF_DEPTH) if opt.camera_scale_method == "pointmap" else None
        print(f"[in] rgbs={tuple(rgbs_in.shape)} c2ws={tuple(c2ws_in.shape)} intr={intr_in[0].tolist()}")

        # 2. Load the GT trajectory: 24 frames from cam_gt, all timesteps.
        gt_traj = ds.get_data(
            idx,
            data_fields=[DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS],
            frame_indices=list(range(F_total)),
            view_indices=[cam_gt],
        )
        rgbs_gt_traj = gt_traj[DF_IMAGE_RGB]            # (F, 3, H, W)
        c2ws_gt_traj = gt_traj[DF_CAMERA_C2W_TRANSFORM] # (F, 4, 4)
        intr_gt = gt_traj[DF_CAMERA_INTRINSICS][0]      # (4,)

        # ----- Apply mean_cam normalization across (inputs + trajectory) jointly -----
        # We want all cameras in the same normalized frame.  Stack input + 24
        # trajectory c2ws and normalize by the mean of the first V_in.
        c2ws_combined = torch.cat([c2ws_in, c2ws_gt_traj], dim=0)  # (V_in + F, 4, 4)
        c2ws_combined = normalize_mean_cam(c2ws_combined, V_in)

        # ----- Pointmap scaling -----
        intrinsics_native = intr_in.clone()  # (V_in, 4); same across cameras (kubric)
        if opt.camera_scale_method == "pointmap":
            scene_scale = 1.0  # matches data_mode='kubric_scaled_1.0'
            # raw_depths shape needed: (V_in, 1, H, W).  Already in this shape via
            # Kubric class which returns (N, 1, H, W) for DF_DEPTH.
            # raw_intrinsics: (V_in, 4) at native resolution.
            avg = compute_pointmap_scale(
                depth_in,
                c2ws_in,
                intr_in,
                V_in,
                trim_lo=getattr(opt, "pointmap_trim_lo", 0.0),
                trim_hi=getattr(opt, "pointmap_trim_hi", 1.0),
            )
            final_scene_scale = scene_scale / avg
        elif opt.camera_scale_method == "distance":
            dist = max(
                torch.max(
                    torch.norm(
                        c2ws_combined[:V_in, :3, 3] - c2ws_combined[0:1, :3, 3], dim=1
                    )
                ).item(),
                1e-6,
            )
            final_scene_scale = 1.0 / dist
        else:
            final_scene_scale = 1.0
        print(f"[scale] camera_scale_method={opt.camera_scale_method} final_scene_scale={final_scene_scale:.4f}")

        c2ws_combined = c2ws_combined.clone()
        c2ws_combined[:, :3, 3] *= final_scene_scale

        c2ws_in_n = c2ws_combined[:V_in]            # (V_in, 4, 4)
        c2ws_traj_n = c2ws_combined[V_in:]           # (F, 4, 4)

        # ----- Image resize to opt.img_size -----
        H_tgt, W_tgt = opt.img_size
        img_tf = ImageTransform(crop_size=(H_tgt, W_tgt), sample_size=(H_tgt, W_tgt), max_crop=True)
        rgbs_in_resized, shift, scale, _ = img_tf.preprocess_images(rgbs_in)  # (V_in, 3, H_tgt, W_tgt)
        rgbs_gt_traj_resized, _, _, _ = img_tf.preprocess_images(rgbs_gt_traj)

        intrinsics_in_resized = torch.stack(
            [
                intr_in[:, 0] * scale[0],
                intr_in[:, 1] * scale[1],
                (intr_in[:, 2] + shift[0]) * scale[0],
                (intr_in[:, 3] + shift[1]) * scale[1],
            ],
            dim=-1,
        )
        intrinsics_traj_resized = torch.stack(
            [
                torch.full((F_total,), intr_gt[0].item()) * scale[0],
                torch.full((F_total,), intr_gt[1].item()) * scale[1],
                (torch.full((F_total,), intr_gt[2].item()) + shift[0]) * scale[0],
                (torch.full((F_total,), intr_gt[3].item()) + shift[1]) * scale[1],
            ],
            dim=-1,
        )

        # ----- Build encoder input (input views) -----
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        )
        images_input_norm = norm(rgbs_in_resized.clone())

        # Per-input time embedding (only the inputs' times, separate from target's).
        # NOTE: in training the time embedding is *joint*-normalized with the
        # target time.  Here we keep the same convention: normalize all candidate
        # times (4 input + F potential targets) together to [0,1].
        all_candidate_times = torch.tensor(
            list(input_frame_indices) + list(range(F_total)), dtype=torch.float32
        )
        t_min = all_candidate_times.min()
        t_max = all_candidate_times.max()
        span = (t_max - t_min).clamp(min=1.0)

        def normt(t_int: int) -> float:
            return ((torch.tensor([float(t_int)]) - t_min) / span).item()

        inp_times_norm = torch.tensor(
            [normt(t) for t in input_frame_indices], dtype=torch.float32
        )
        inp_time_emb = timestep_embedding(inp_times_norm, opt.time_embedding_dim)  # (V_in, T_dim)
        inp_time_emb_full = inp_time_emb[..., None, None] * torch.ones_like(images_input_norm[:, :1])
        encoder_input_image = torch.cat([images_input_norm, inp_time_emb_full], dim=1)  # (V_in, 3+T_dim, H, W)

        # plucker + rays for input views
        plucker_in, rays_os_in, rays_ds_in = ray_condition(
            intrinsics_in_resized[None],
            c2ws_in_n[None],
            H_tgt,
            W_tgt,
            device="cpu",
        )

        # Move everything to GPU for inference
        images_rgb_in = rgbs_in_resized[None].to(device)            # (B=1, V_in, 3, H, W)
        plucker_in = plucker_in[None].to(device)                    # (1, V_in, 6, H, W)
        rays_os_in = rays_os_in[None].to(device)
        rays_ds_in = rays_ds_in[None].to(device)
        intrinsics_input_t = intrinsics_in_resized[None].to(device)
        cam_to_world_input_t = c2ws_in_n[None].to(device)
        time_embedding_input_t = inp_time_emb_full[None].to(device)  # (1, V_in, T_dim, H, W)
        # Renormalize the encoder image stack (3 channels, not 3+T_dim) for the
        # encoder's images_rgb input; the model expects RGB only there.
        images_rgb_in_norm = norm(rgbs_in_resized[None].clone()).to(device)

        encoder_input = ModelInputEncoder(
            images_rgb=images_rgb_in_norm,
            plucker=plucker_in,
            rays_os=rays_os_in,
            rays_ds=rays_ds_in,
            intrinsics_input=intrinsics_input_t,
            cam_to_world_input=cam_to_world_input_t,
            time_embedding_input=time_embedding_input_t,
            time_embedding_target=None,  # filled per-frame below
            images_rgb_unnormalized=rgbs_in_resized[None].to(device),
        )

        # ----- Encode once -----
        print("[encode] running encoder...")
        encoder_latent = model.forward_encoder(encoder_input)
        print(f"[encode] keys={tuple(encoder_latent.keys.shape)} values={tuple(encoder_latent.values.shape)}")

        # ----- Helper: render gaussians at a single (cam_view, intrinsics, target_time) -----
        def render_at(
            c2w_target: torch.Tensor,      # (4, 4)
            intr_target: torch.Tensor,      # (4,)
            t_int: int,
            dynamic_only: bool = False,
            static_only: bool = False,
        ) -> np.ndarray:
            # Build single-view decoder input.
            c2w_target_b = c2w_target[None, None].to(device)  # (1, 1, 4, 4)
            cam_view_target = torch.inverse(c2w_target_b).transpose(-2, -1)  # (1, 1, 4, 4)
            intr_target_b = intr_target[None, None].to(device)  # (1, 1, 4)

            t_norm = normt(t_int)
            tgt_time_emb_1d = timestep_embedding(
                torch.tensor([t_norm], dtype=torch.float32), opt.time_embedding_dim
            )  # (1, T_dim)
            # Decoder time embedding shape per provider: (B=1, V_out=1, T_dim, H_tgt, W_tgt)
            tgt_time_emb = (
                tgt_time_emb_1d[..., None, None]
                * torch.ones(1, 1, opt.time_embedding_dim, H_tgt, W_tgt)
            ).to(device)

            decoder_input = ModelInputDecoder(
                time_embedding_target=tgt_time_emb,
                cam_view=cam_view_target,
                intrinsics=intr_target_b,
            )

            gaussians = model.forward_decoder(encoder_latent, decoder_input)  # (1, N*P^2, 14)
            # The activation head expands each token into dec_patch_size^2 = P^2
            # gaussians; total = (num_gs_tokens + num_dynamic_gs_tokens) * P^2.
            # Static block: FIRST num_gs_tokens * P^2 entries.
            # Dynamic block: LAST num_dynamic_gs_tokens * P^2 entries.
            gs_per_token = opt.dec_patch_size * opt.dec_patch_size
            if dynamic_only:
                n_dyn_gs = opt.num_dynamic_gs_tokens * gs_per_token
                gaussians = gaussians[:, -n_dyn_gs:, :].contiguous()
            elif static_only:
                n_static_gs = opt.num_gs_tokens * gs_per_token
                gaussians = gaussians[:, :n_static_gs, :].contiguous()

            results = model.render_gaussians(gaussians, decoder_input)
            img = results["images_pred"][0, 0]  # (3, H, W)
            img = img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            return (img * 255).astype(np.uint8)

        # ----- Video 1: GT input video (no inference) -----
        print("[v1] saving GT input video...")
        gt_frames = (
            rgbs_gt_traj_resized.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
        ).astype(np.uint8)
        write_mp4(gt_frames, os.path.join(scene_out_dir, "01_gt_input.mp4"), fps=args.fps)

        # ----- Video 2: render along GT trajectory, varying time -----
        print("[v2] rendering along GT trajectory ({} frames)...".format(F_total))
        v2_frames = []
        for t in range(F_total):
            img = render_at(c2ws_traj_n[t], intrinsics_traj_resized[t], t)
            v2_frames.append(img)
        v2 = np.stack(v2_frames, axis=0)
        write_mp4(v2, os.path.join(scene_out_dir, "02_gt_trajectory.mp4"), fps=args.fps)

        # ----- Video 3: render from frame-0 camera, varying time -----
        print("[03_fixed_cam] rendering from fixed (frame-0) camera, varying time...")
        fixed_c2w = c2ws_traj_n[0]
        fixed_intr = intrinsics_traj_resized[0]
        fixed_cam_frames = []
        for t in range(F_total):
            img = render_at(fixed_c2w, fixed_intr, t)
            fixed_cam_frames.append(img)
        fixed_cam = np.stack(fixed_cam_frames, axis=0)
        write_mp4(fixed_cam, os.path.join(scene_out_dir, "03_fixed_cam.mp4"), fps=args.fps)

        # ----- Video 4: dynamic gaussians only, same camera, varying time -----
        if opt.num_dynamic_gs_tokens > 0:
            print("[v4] rendering dynamic-tokens-only from fixed camera, varying time...")
            v4_frames = []
            for t in range(F_total):
                img = render_at(fixed_c2w, fixed_intr, t, dynamic_only=True)
                v4_frames.append(img)
            v4 = np.stack(v4_frames, axis=0)
            write_mp4(v4, os.path.join(scene_out_dir, "04_fixed_cam_dyn_only.mp4"), fps=args.fps)

        # ----- Video 5: static gaussians only, same camera, varying time -----
        if opt.num_gs_tokens > 0:
            print("[v5] rendering static-tokens-only from fixed camera, varying time...")
            v5_frames = []
            for t in range(F_total):
                img = render_at(fixed_c2w, fixed_intr, t, static_only=True)
                v5_frames.append(img)
            v5 = np.stack(v5_frames, axis=0)
            write_mp4(v5, os.path.join(scene_out_dir, "05_fixed_cam_static_only.mp4"), fps=args.fps)

        # ----- Video 6/7/8: hstack renders from each input camera, varying time -----
        # For each of the V_in input cameras, render F_total frames with time
        # varying. Concatenate horizontally per frame to produce a [H, V_in*W]
        # image, then stack temporally into an mp4.
        def render_hstack(label: str, fname: str, *, dynamic_only: bool = False, static_only: bool = False):
            print(f"[{label}] rendering input-cam hstack from {V_in} cams, varying time...")
            per_frame = []
            for t in range(F_total):
                cols = []
                for v in range(V_in):
                    img = render_at(
                        c2ws_in_n[v], intrinsics_in_resized[v], t,
                        dynamic_only=dynamic_only, static_only=static_only,
                    )
                    cols.append(img)
                per_frame.append(np.concatenate(cols, axis=1))  # hstack along width
            vid = np.stack(per_frame, axis=0)
            write_mp4(vid, os.path.join(scene_out_dir, fname), fps=args.fps)

        render_hstack("v6", "06_input_cams_hstack.mp4")
        if opt.num_dynamic_gs_tokens > 0:
            render_hstack("v7", "07_input_cams_hstack_dyn.mp4", dynamic_only=True)
        if opt.num_gs_tokens > 0:
            render_hstack("v8", "08_input_cams_hstack_static.mp4", static_only=True)

        print("[done] wrote videos to:", scene_out_dir)


if __name__ == "__main__":
    main()
