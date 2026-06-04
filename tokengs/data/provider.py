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

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import inspect

from tokengs.utils.data import ImageTransform, ray_condition, timestep_embedding
from tokengs.options import Options
from tokengs.data.registry import dataset_registry
from tokengs.utils.augmentation import random_reflect
from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_IMAGE_RGB,
    DF_FOREGROUND_MASK,
    DF_DEPTH,
)


def _accepts_kwarg(sig: inspect.Signature, name: str) -> bool:
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


class Provider(Dataset):
    def __init__(self, dataset_name: str, opt: Options, training: bool = True, num_repeat: int = 1):
        self.opt = opt
        if '_scaled_' in dataset_name:
            # overwrite the scene scale
            scale_factor = float(dataset_name.split('_scaled_')[-1])
            dataset_name = dataset_name.split('_scaled_')[0]
            override_scene_scale = scale_factor
        else:
            override_scene_scale = None
        dataset_entry = dataset_registry[dataset_name]
        dataset_kwargs = dict(dataset_entry['kwargs'])
        if opt.camera_scale_method == 'pointmap':
            init_sig = inspect.signature(dataset_entry['cls'].__init__)
            if _accepts_kwarg(init_sig, "load_depth"):
                dataset_kwargs['load_depth'] = True
        if opt.dataset_kwargs is not None:
            dataset_kwargs.update(opt.dataset_kwargs)
        self.dataset = dataset_entry['cls'](**dataset_kwargs)
        self._get_data_accepts_num_depth_frames = _accepts_kwarg(
            inspect.signature(self.dataset.get_data),
            "num_depth_frames",
        )
        self.scene_scale = dataset_entry['scene_scale'] if override_scene_scale is None else override_scene_scale
        self.max_gap, self.min_gap = dataset_entry['max_gap'], dataset_entry['min_gap']
        self.training = training
        self.dataset.sample_list *= num_repeat

        if not opt.evaluating:
            if training:
                self.dataset.sample_list = self.dataset.sample_list[:-self.opt.batch_size]
            else:
                self.dataset.sample_list = self.dataset.sample_list[-self.opt.batch_size:]

        self.rng = np.random.default_rng(self.opt.seed)
        self.generator = torch.Generator(device='cpu').manual_seed(self.opt.seed)

        self._setup_image_transforms(
            sample_size=self.opt.img_size,
            crop_size=self.opt.img_size,
            max_crop=True,
        )

        self.data_fields = [DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS, DF_FOREGROUND_MASK]
        if opt.camera_scale_method == 'pointmap':
            self.data_fields.append(DF_DEPTH)

    def set_rng_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(epoch + self.opt.seed)
        self.generator = torch.Generator(device='cpu').manual_seed(epoch + self.opt.seed)

    def __len__(self):
        return len(self.dataset)

    def _setup_image_transforms(self, sample_size, crop_size, max_crop=False):
        self.image_transform = ImageTransform(
            crop_size=crop_size, sample_size=sample_size, max_crop=max_crop
        )
        self.input_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

    def _normalize_camera_mean_cam(self, c2ws):
        input_c2ws = c2ws[:self.opt.num_input_views]
        # normalize input camera poses
        position_avg = input_c2ws[:, :3, 3].mean(0)  # (3,)
        forward_avg = input_c2ws[:, :3, 2].mean(0)  # (3,)
        down_avg = input_c2ws[:, :3, 1].mean(0)  # (3,)
        # gram-schmidt process
        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)
        pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)  # (3, 4)
        pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0)  # (4, 4)
        pos_avg_inv = torch.inverse(pos_avg)

        c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), c2ws)
        return c2ws

    def _compute_pointmap_scale(self, raw_depths, c2ws, raw_intrinsics):
        """Normalize cameras by mean input point distance from the origin."""
        V = min(self.opt.num_input_views, len(raw_depths))
        all_dists = []
        for i in range(V):
            depth = raw_depths[i, 0]
            valid = depth > 1e-8
            if not valid.any():
                continue
            fx, fy, cx, cy = raw_intrinsics[i]
            H, W = depth.shape
            u = torch.arange(W, dtype=torch.float32)
            v = torch.arange(H, dtype=torch.float32)
            uu, vv = torch.meshgrid(u, v, indexing='xy')
            x_cam = (uu - cx) / fx * depth
            y_cam = (vv - cy) / fy * depth
            z_cam = depth
            pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
            R = c2ws[i, :3, :3]
            t = c2ws[i, :3, 3]
            pts_world = pts_cam[valid] @ R.T + t
            all_dists.append(pts_world.norm(dim=-1))
        if len(all_dists) == 0:
            return 1.0
        all_dists = torch.cat(all_dists, dim=0)

        lo = float(getattr(self.opt, "pointmap_trim_lo", 0.0))
        hi = float(getattr(self.opt, "pointmap_trim_hi", 1.0))
        if not (0.0 <= lo < hi <= 1.0):
            lo, hi = 0.0, 1.0
        if (lo > 0.0 or hi < 1.0) and all_dists.numel() > 1:
            q_lo = torch.quantile(all_dists, lo) if lo > 0.0 else all_dists.min()
            q_hi = torch.quantile(all_dists, hi) if hi < 1.0 else all_dists.max()
            trimmed = all_dists[(all_dists >= q_lo) & (all_dists <= q_hi)]
            if trimmed.numel() > 0:
                all_dists = trimmed

        return all_dists.mean().clamp(min=1e-6).item()

    def _preprocess(self, rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask):
        if self.opt.camera_scale_method == 'pointmap':
            raw_depths = depths.clone()
            raw_intrinsics = intrinsics.clone()

        rgbs, shift, scale, flip_flag = self.image_transform.preprocess_images(rgbs)
        masks, _, _, _ = self.image_transform.preprocess_images(masks)
        depths, _, _, _ = self.image_transform.preprocess_images(depths)
        intrinsics = torch.stack(
            [
                intrinsics[..., 0] * scale[0],
                intrinsics[..., 1] * scale[1],
                (intrinsics[..., 2] + shift[0]) * scale[0],
                (intrinsics[..., 3] + shift[1]) * scale[1],
            ],
            dim=-1,
        )

        # relative pose
        if self.opt.camera_normalization_method == 'mean_cam':
            c2ws = self._normalize_camera_mean_cam(c2ws)
        elif self.opt.camera_normalization_method == 'first_cam':
            c2ws = torch.inverse(c2ws[0]).unsqueeze(0) @ c2ws
        else:
            raise ValueError(f"Invalid camera normalization method: {self.opt.camera_normalization_method}")

        # compute the scene scale
        if self.opt.camera_scale_method == 'constant':
            final_scene_scale = self.scene_scale
        elif self.opt.camera_scale_method == 'distance':
            dist = max(
                torch.max(torch.norm(c2ws[: self.opt.num_input_views, :3, 3] - c2ws[0:1, :3, 3], dim=1)),
                1e-6,
            )
            final_scene_scale = self.scene_scale/dist
        elif self.opt.camera_scale_method == 'bound':
            position_max = c2ws[: self.opt.num_input_views, :3, 3].abs().max()
            final_scene_scale = self.scene_scale/position_max
        elif self.opt.camera_scale_method == 'pointmap':
            avg_dist = self._compute_pointmap_scale(raw_depths, c2ws, raw_intrinsics)
            final_scene_scale = self.scene_scale / avg_dist
        else:
            raise ValueError(f"Invalid camera scale method: {self.opt.camera_scale_method}")

        if self.training and self.opt.random_reflect:
            rgbs, c2ws = random_reflect(rgbs, c2ws, generator=self.generator)

        c2ws[:, :3, 3] = c2ws[:, :3, 3] * final_scene_scale

        images_input = self.input_normalizer(rgbs.clone())
        if self.opt.time_embedding:
            timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())
            time_embeddings = timestep_embedding(timesteps, self.opt.time_embedding_dim)
            time_embeddings = time_embeddings[..., None, None] * torch.ones_like(images_input[:, :1])
            images_input = torch.cat([images_input, time_embeddings], dim=1)

        plucker_embedding, rays_os, rays_ds = ray_condition(
            intrinsics[None],
            c2ws[None],
            self.opt.img_size[0],
            self.opt.img_size[1],
            device="cpu",
            flip_flag=flip_flag,
        )
        final_input = torch.cat([images_input, plucker_embedding], dim=1)

        return {
            'input': final_input,
            'rays_os': rays_os,
            'rays_ds': rays_ds,
            "images_all": rgbs,
            'images_input': rgbs[:self.opt.num_input_views],
            'images_output': rgbs[self.opt.num_input_views:],
            "intrinsics_all": intrinsics,
            'intrinsics': intrinsics[self.opt.num_input_views:],
            'intrinsics_input': intrinsics[:self.opt.num_input_views],
            'cam_view_all': torch.inverse(c2ws).transpose(1, 2), # [V, 4, 4]
            'cam_view': torch.inverse(c2ws[self.opt.num_input_views:]).transpose(1, 2), # [V, 4, 4]
            'masks_all': masks,
            'masks_output': masks[self.opt.num_input_views:],
            'has_mask': has_mask,
            'cam_to_world_input': c2ws[:self.opt.num_input_views], # [V, 4, 4]
            'cam_view_input': torch.inverse(c2ws[:self.opt.num_input_views]).transpose(1, 2), # [V, 4, 4]
        }
    
    def get_rng(self, idx: int) -> np.random.Generator:
        """
        Get a random number generator for the given index.
        If training, use a shared RNG to produce different samples across batches.
        If eval, use a fixed RNG for a given index to produce the same sample across evaluations.
        """
        if self.training:
            return self.rng
        else:
            return np.random.default_rng(self.opt.seed + idx)

    def _get_indices_dynamic(self, idx):
        rng = self.get_rng(idx)
        total_num_frames = self.dataset.count_frames(idx)
        camera_count = self.dataset.count_cameras(idx)
        assert total_num_frames >= self.opt.num_input_views, f'Frame number {total_num_frames} is smaller than number of input views {self.opt.num_input_views}.'
        context_gap = rng.integers(self.min_gap, self.max_gap + 1)
        context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views - 1)

        start_frame = rng.integers(0, total_num_frames-context_gap)
        inbetween_indices = np.sort(rng.permutation(np.arange(start_frame + 1, start_frame+context_gap))[:self.opt.num_input_views - 2])
        frame_indices = np.array([start_frame, *inbetween_indices, start_frame+context_gap])
        target_index = rng.permutation(np.arange(start_frame, start_frame+context_gap+1))[:1]

        if not self.opt.use_interp_target:
            target_index = rng.permutation(frame_indices)[:1]

        # append to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        if self.opt.num_input_views > camera_count:
            view_indices = rng.permutation(np.arange(self.opt.num_input_views)%camera_count)
        else:
            view_indices = rng.permutation(np.arange(camera_count))[:self.opt.num_input_views]
        
        return frame_indices, view_indices

    def _get_indices_static(self, idx):
        rng = self.get_rng(idx)

        total_num_frames = self.dataset.count_frames(idx)
        assert total_num_frames >= max(self.opt.num_input_views, self.opt.num_views - self.opt.num_input_views), f'Frame number {total_num_frames} is smaller than number of input views {max(self.opt.num_input_views, self.opt.num_views - self.opt.num_input_views)}.'
        context_gap = rng.integers(self.min_gap, self.max_gap + 1)
        context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views - 1)
        start_frame = rng.integers(0, total_num_frames-context_gap)
        inbetween_indices = np.sort(rng.permutation(np.arange(start_frame + 1, start_frame+context_gap))[:self.opt.num_input_views - 2])
        frame_indices = np.array([start_frame, *inbetween_indices, start_frame+context_gap])
        target_index = rng.permutation(np.arange(start_frame, start_frame+context_gap+1))[:self.opt.num_views-self.opt.num_input_views]

        # append to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])
        
        return frame_indices, []

    def _get_indices_eval(self, idx):
        """
        Get indices for evaluation datasets that have predefined context and target frames.
        Uses the get_context_target_frames method from the dataset.
        """
        context_frames, target_frames = self.dataset.get_context_target_frames(idx)
        frame_indices = np.concatenate([np.asarray(context_frames), np.asarray(target_frames)])
        return frame_indices, []

    def _curate_batch_static(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        return all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices

    def _curate_batch_dynamic(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        rgbs, masks, depths, c2ws, intrinsics, timesteps = [], [], [], [], [], []
        v_in = self.opt.num_input_views
        grid_shape = [v_in, v_in + 1]
        all_rgbs = all_rgbs.reshape([*grid_shape, *all_rgbs.shape[1:]])
        all_masks = all_masks.reshape([*grid_shape, *all_masks.shape[1:]])
        all_depths = all_depths.reshape([*grid_shape, *all_depths.shape[1:]])
        all_c2ws = all_c2ws.reshape([*grid_shape, *all_c2ws.shape[1:]])
        all_intrinsics = all_intrinsics.reshape([*grid_shape, *all_intrinsics.shape[1:]])
        # input views
        for v in range(self.opt.num_input_views):
            rgbs.append(all_rgbs[v, v])
            masks.append(all_masks[v, v])
            depths.append(all_depths[v, v])
            c2ws.append(all_c2ws[v, v])
            intrinsics.append(all_intrinsics[v, v])
            timesteps.append(frame_indices[v])
        # supervision views
        assert self.opt.num_views <= self.opt.num_input_views * 2, f"Total views should be less than twice of input views {self.opt.num_input_views}, instead got {self.opt.num_views}"
        for v in range(self.opt.num_views - self.opt.num_input_views):
            rgbs.append(all_rgbs[v, -1])
            masks.append(all_masks[v, -1])
            depths.append(all_depths[v, -1])
            c2ws.append(all_c2ws[v, -1])
            intrinsics.append(all_intrinsics[v, -1])
            timesteps.append(frame_indices[-1])
        rgbs, masks, depths, c2ws, intrinsics, timesteps = torch.stack(rgbs), torch.stack(masks), torch.stack(depths), torch.stack(c2ws), torch.stack(intrinsics), torch.stack(timesteps)
        return rgbs, masks, depths, c2ws, intrinsics, timesteps

    def _curate_batch_dynamic_eval(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        """
        Curate batch for dynamic evaluation datasets (e.g., DyCheckMVEval).
        
        For evaluation:
        - First num_input_views frames are context (from one camera, specific timesteps)
        - Remaining frames are targets (from other cameras, various timesteps)
        - No grid structure needed, data is already in the correct order
        """
        # Split context and target
        num_context = self.opt.num_input_views
        
        # Context frames
        rgbs_context = all_rgbs[:num_context]
        masks_context = all_masks[:num_context]
        depths_context = all_depths[:num_context]
        c2ws_context = all_c2ws[:num_context]
        intrinsics_context = all_intrinsics[:num_context]
        timesteps_context = frame_indices[:num_context]
        
        # Target frames
        rgbs_target = all_rgbs[num_context:]
        masks_target = all_masks[num_context:]
        depths_target = all_depths[num_context:]
        c2ws_target = all_c2ws[num_context:]
        intrinsics_target = all_intrinsics[num_context:]
        timesteps_target = frame_indices[num_context:]
        
        # Concatenate context and target
        rgbs = torch.cat([rgbs_context, rgbs_target], dim=0)
        masks = torch.cat([masks_context, masks_target], dim=0)
        depths = torch.cat([depths_context, depths_target], dim=0)
        c2ws = torch.cat([c2ws_context, c2ws_target], dim=0)
        intrinsics = torch.cat([intrinsics_context, intrinsics_target], dim=0)
        timesteps = torch.cat([timesteps_context, timesteps_target], dim=0)
        
        return rgbs, masks, depths, c2ws, intrinsics, timesteps

    def get_item(self, idx):
        if hasattr(self.dataset, 'get_context_target_frames'):
            _get_indices_fn = self._get_indices_eval
            # Use special curate function for dynamic evaluation datasets
            _curate_batch_fn = self._curate_batch_static if self.dataset.is_static else self._curate_batch_dynamic_eval
        else:
            _get_indices_fn = self._get_indices_static if self.dataset.is_static else self._get_indices_dynamic
            _curate_batch_fn = self._curate_batch_static if self.dataset.is_static else self._curate_batch_dynamic

        frame_indices, view_indices = _get_indices_fn(idx)
        extra_kwargs = {}
        if self.opt.camera_scale_method == 'pointmap' and self._get_data_accepts_num_depth_frames:
            extra_kwargs['num_depth_frames'] = self.opt.num_input_views
        original_output_dict = self.dataset.get_data(
            idx,
            data_fields=self.data_fields,
            frame_indices=frame_indices,
            view_indices=view_indices,
            **extra_kwargs,
        )

        if not (has_mask := (DF_FOREGROUND_MASK in original_output_dict)):
            original_output_dict[DF_FOREGROUND_MASK] = torch.ones_like(
                original_output_dict[DF_IMAGE_RGB][:, 0:1, ...]
            )
        if DF_DEPTH not in original_output_dict:
            original_output_dict[DF_DEPTH] = torch.ones_like(original_output_dict[DF_IMAGE_RGB][:, 0:1, ...])

        all_rgbs, all_c2ws, all_intrinsics, all_masks, all_depths = original_output_dict[DF_IMAGE_RGB], original_output_dict[DF_CAMERA_C2W_TRANSFORM], original_output_dict[DF_CAMERA_INTRINSICS], original_output_dict[DF_FOREGROUND_MASK], original_output_dict[DF_DEPTH]

        rgbs, masks, depths, c2ws, intrinsics, timesteps = _curate_batch_fn(all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, torch.from_numpy(frame_indices).float())

        return self._preprocess(rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask)

    def __getitem__(self, idx):
        while True:
            try:
                return self.get_item(idx)
            except Exception:
                idx = self.rng.integers(0, len(self.dataset))
