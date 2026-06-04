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

import json
import struct
import zipfile
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
from pathlib import Path
import glob
from PIL import Image
from io import BytesIO

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_IMAGE_RGB,
)


class DL3DV10K:
    def __init__(self, root_path, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', load_depth=False, **kwargs):
        """
        data_format: support different formats for what different code base expect
        """
        super().__init__(**kwargs)
        self.root_path = root_path
        self.subset = subset
        self.load_depth = load_depth

        self.sample_list = []
        for sub in self.subset:
            if sub == '140':
                cur_sample_list = sorted(glob.glob(f"{root_path}/*.zip"))
            else:
                cur_sample_list = sorted(glob.glob(f"{root_path}/{sub}/*.zip"))
            self.sample_list.extend(cur_sample_list)

        if resolution == '960p':
            self.resolution = [540, 960]
            self.image_folder = 'images_4'
        elif resolution == '960p_images':
            # For training dataset that uses 'images' folder instead of 'images_4'
            self.resolution = [540, 960]
            self.image_folder = 'images'
        else:
            raise NotImplementedError(f"Resolution {resolution} not supported")
        
        self.is_static = True

    def __len__(self):
        return len(self.sample_list)
    
    def load_intrinsics(self, data_dict, resolution = None):
        fx = data_dict['fl_x']
        fy = data_dict['fl_y']
        cx = data_dict['cx']
        cy = data_dict['cy']

        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        if resolution is not None:
            H_new, W_new = resolution
            W_old = data_dict['w']
            H_old = data_dict['h']

            # Scale the intrinsics to the new resolution
            intrinsics[0] *= W_new / W_old
            intrinsics[1] *= H_new / H_old
            intrinsics[2] *= W_new / W_old
            intrinsics[3] *= H_new / H_old

        return intrinsics

    def load_video_reader(self, idx):
        zip_path = self.sample_list[idx]
        zip_handle = zipfile.ZipFile(zip_path, "r")

        clip_name = Path(zip_path).stem

        if self.subset == ['140']:
            clip_name = f"{clip_name}/gaussian_splat"

        # load json data
        with zip_handle.open(f"{clip_name}/transforms.json", "r") as f:
            json_data = json.load(f)

        # load video length
        video_length = len(json_data['frames'])

        # load intrinsics
        intrinsics = self.load_intrinsics(json_data, resolution = self.resolution)
        transform_matrix_all = self.load_cameras(json_data)

        return clip_name, video_length, intrinsics, transform_matrix_all, zip_handle, json_data

    def load_cameras(self, data_dict):
        transform_matrix_all = []
        for frame_data in data_dict['frames']:
            transform_matrix = np.array(frame_data['transform_matrix'])
            c2w = transform_matrix
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            transform_matrix_all.append(c2w)
        return np.stack(transform_matrix_all, axis=0)

    def count_cameras(self, video_idx: int) -> int:
        return 1
    
    def count_frames(self, idx):
        zip_path = self.sample_list[idx]
        with zipfile.ZipFile(zip_path, "r") as zip_handle:
            clip_name = Path(zip_path).stem
            if self.subset == ['140']:
                clip_name = f"{clip_name}/gaussian_splat"
            # load json data
            with zip_handle.open(f"{clip_name}/transforms.json", "r") as f:
                json_data = json.load(f)
            total_frames = len(json_data['frames'])

        return total_frames

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = "opencv",
        num_depth_frames: Optional[int] = None,
    ):
        assert camera_convention == "opencv"

        clip_name, total_frames, intrinsics, transform_matrices, zip_handle, json_data = self.load_video_reader(idx)
        if frame_indices is None:
            frame_indices = range(total_frames)

        # load camera poses
        c2w = transform_matrices[frame_indices]
        
        # load img_seq
        img_seq = []
        for frame_idx in frame_indices:
            img_name = json_data['frames'][frame_idx]['file_path'].split('/')[-1]
            with zip_handle.open(f"{clip_name}/{self.image_folder}/{img_name}", "r") as f:
                img = Image.open(BytesIO(f.read()))
                img_seq.append(np.array(img))
        img_seq = np.stack(img_seq, axis=0) # n h w c
        img_seq = torch.from_numpy(img_seq).permute(0, 3, 1, 2).contiguous()
        img_seq = img_seq / 255.0  # (0,1)

        c2w = torch.from_numpy(c2w).float().contiguous()
        # prepare intrinsics
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).repeat(len(frame_indices), 1)

        output_dict = {}
        output_dict["__key__"] = clip_name
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                output_dict[data_field] = img_seq
            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                output_dict[data_field] = c2w
            elif data_field == DF_CAMERA_INTRINSICS:
                output_dict[data_field] = intrinsics
            elif data_field == DF_DEPTH and self.load_depth:
                depth_seq = self._load_depth_seq(
                    clip_name, json_data, zip_handle, frame_indices, num_depth_frames
                )
                output_dict[data_field] = torch.from_numpy(depth_seq).float().unsqueeze(1)


        return output_dict

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        zip_handle,
        frame_indices,
        num_depth_frames: Optional[int],
    ) -> np.ndarray:
        n_load = len(frame_indices) if num_depth_frames is None else min(num_depth_frames, len(frame_indices))
        depth_seq = []
        for i, frame_idx in enumerate(frame_indices):
            if i < n_load:
                depth_path = f"{clip_name}/{json_data['frames'][frame_idx]['depth_path']}"
                with zip_handle.open(depth_path, "r") as f:
                    depth = np.load(BytesIO(f.read()))
            else:
                depth = np.zeros(self.resolution, dtype=np.float64)
            depth_seq.append(depth)
        return np.stack(depth_seq, axis=0)


class DL3DVEval(DL3DV10K):
    def __init__(self, root_path, evaluation_json, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', num_input=16, load_depth=False):
        super().__init__(root_path, subset, resolution, load_depth=load_depth)

        self.evaluation_indices = json.load(open(evaluation_json, "r"))
        self.sample_list = []
        for k in self.evaluation_indices:
            if isinstance(k, str):
                self.sample_list.append(os.path.join(root_path, k+'.zip'))
            else:
                self.sample_list.append(os.path.join(root_path, k['scene_name']+'.zip'))
        
        self.is_static = True

        self.num_input = num_input
        self._colmap_points_cache: Dict[str, Tuple[np.ndarray, List[frozenset]]] = {}

    def _read_points3d_bin(
        self, zip_handle: zipfile.ZipFile, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        path = f"{clip_name}/sparse/0/points3D.bin"
        with zip_handle.open(path, "r") as f:
            data = f.read()

        off = 0
        (n_points,) = struct.unpack_from("<Q", data, off)
        off += 8
        xyz = np.empty((n_points, 3), dtype=np.float64)
        image_id_sets: List[frozenset] = []
        for i in range(n_points):
            off += 8
            xyz[i] = struct.unpack_from("<ddd", data, off)
            off += 24 + 3 + 8
            (track_len,) = struct.unpack_from("<Q", data, off)
            off += 8
            track = struct.unpack_from("<" + "II" * track_len, data, off)
            off += 8 * track_len
            image_id_sets.append(frozenset(track[::2]))
        return xyz, image_id_sets

    def _get_colmap_points(
        self, zip_handle: zipfile.ZipFile, clip_name: str
    ) -> Tuple[np.ndarray, List[frozenset]]:
        cached = self._colmap_points_cache.get(clip_name)
        if cached is not None:
            return cached
        cached = self._read_points3d_bin(zip_handle, clip_name)
        self._colmap_points_cache[clip_name] = cached
        return cached

    @staticmethod
    def _colmap_world_to_final_world_transform(json_data: Dict) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        applied = json_data.get("applied_transform")
        if applied is not None:
            transform[:3, :] = np.asarray(applied, dtype=np.float64)

        z_flip = np.diag([1.0, 1.0, -1.0, 1.0])
        xy_swap = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        return xy_swap @ z_flip @ transform

    def _load_depth_seq(
        self,
        clip_name,
        json_data,
        zip_handle,
        frame_indices,
        num_depth_frames: Optional[int],
    ) -> np.ndarray:
        xyz_all, image_id_sets = self._get_colmap_points(zip_handle, clip_name)
        world_transform = self._colmap_world_to_final_world_transform(json_data)

        intrinsics = self.load_intrinsics(json_data, resolution=self.resolution)
        fx, fy, cx, cy = intrinsics
        c2ws_all = self.load_cameras(json_data)
        H, W = self.resolution

        xyz_h = np.concatenate(
            [xyz_all, np.ones((xyz_all.shape[0], 1), dtype=np.float64)], axis=1
        )
        xyz_world = (xyz_h @ world_transform.T)[:, :3]

        n_frames = len(frame_indices)
        n_load = n_frames if num_depth_frames is None else min(num_depth_frames, n_frames)
        depth_seq = np.zeros((n_frames, H, W), dtype=np.float64)
        for i in range(n_load):
            frame_idx = int(frame_indices[i])
            colmap_image_id = json_data["frames"][frame_idx].get("colmap_im_id")
            if colmap_image_id is None:
                continue

            mask = np.fromiter(
                (colmap_image_id in s for s in image_id_sets),
                dtype=bool,
                count=len(image_id_sets),
            )
            if not mask.any():
                continue

            c2w = c2ws_all[frame_idx]
            w2c = np.linalg.inv(c2w)
            pts_cam = xyz_world[mask] @ w2c[:3, :3].T + w2c[:3, 3]
            z = pts_cam[:, 2]
            in_front = z > 1e-6
            if not in_front.any():
                continue
            pts_cam = pts_cam[in_front]
            z = z[in_front]

            u = fx * pts_cam[:, 0] / z + cx
            v = fy * pts_cam[:, 1] / z + cy
            u_i = np.round(u).astype(np.int64)
            v_i = np.round(v).astype(np.int64)
            in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
            if not in_bounds.any():
                continue

            u_i = u_i[in_bounds]
            v_i = v_i[in_bounds]
            z = z[in_bounds]
            order = np.argsort(-z)
            depth_seq[i, v_i[order], u_i[order]] = z[order]

        return depth_seq

    def get_context_target_frames(self, idx):
        if isinstance(self.evaluation_indices, dict):
            scene_name = Path(self.sample_list[idx]).stem
            eval_data = self.evaluation_indices[scene_name]
            context_frames = eval_data["context"]
            target_frames = eval_data["target"]
            return context_frames, target_frames
        else:
            # llrm eval
            eval_data = self.evaluation_indices[idx]
            if self.num_input == 16:
                context_frames = eval_data["fold_8_kmeans_16_input"]
            elif self.num_input == 32:
                context_frames = eval_data["fold_8_kmeans_32_input"]
            else:
                raise ValueError(f"Unsupported number of input frames: {self.num_input}")
            target_frames = [x for x in range(self.count_frames(idx))]
            return context_frames, target_frames
