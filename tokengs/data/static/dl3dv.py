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
import zipfile
from typing import List, Optional
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
    DF_IMAGE_RGB,
)


class DL3DV10K:
    def __init__(self, root_path, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', **kwargs):
        """
        data_format: support different formats for what different code base expect
        """
        super().__init__(**kwargs)
        self.root_path = root_path
        self.subset = subset

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


        return output_dict


class DL3DVEval(DL3DV10K):
    def __init__(self, root_path, evaluation_json, subset = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], resolution = '960p', num_input=16):
        super().__init__(root_path, subset, resolution)

        self.evaluation_indices = json.load(open(evaluation_json, "r"))
        self.sample_list = []
        for k in self.evaluation_indices:
            if isinstance(k, str):
                self.sample_list.append(os.path.join(root_path, k+'.zip'))
            else:
                self.sample_list.append(os.path.join(root_path, k['scene_name']+'.zip'))
        
        self.is_static = True

        self.num_input = num_input

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