# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Kubric multi-view 4D dataset (dynamic).

Each scene is a directory containing per-camera tar files
(``output_{view_idx:03d}.tar``). Each tar bundles a ``metadata.json``
describing intrinsics + per-frame camera positions/quaternions and
PNG frames named ``rgba_{frame_idx:05d}.png`` (plus optional
``depth_{frame_idx:05d}.tiff`` for pointmap-based scaling).

Adapted to match the ``DL3DV10K``-style ``get_data(...)`` interface used by
``tokengs/data/provider.py`` (returns a dict keyed by ``DF_*`` constants
with tensors instead of a raw-dict + datafield API).
"""

import json
import os
import tarfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from tokengs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_IMAGE_RGB,
)


def _quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix.

    Matches the pyquaternion / Hamilton convention used by the Kubric data.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


class Kubric:
    """Multi-view dynamic Kubric scenes.

    Directory layout (under ``root_path``):
        <root>/<split>/<scene>/output_{view:03d}.tar
    Each tar has:
        - ``metadata.json``  (intrinsics + per-frame camera pos/quat)
        - ``rgba_{frame:05d}.png``
        - ``depth_{frame:05d}.tiff`` (used when ``load_depth=True``)

    Defaults to ``count_frames=24`` and ``count_cameras=9`` matching the
    released Kubric4D multi-view layout.
    """

    DEFAULT_SUBSET: tuple = ("v0", "v1", "v2")

    def __init__(
        self,
        root_path: str,
        subset: Optional[Sequence[str]] = None,
        num_frames: int = 24,
        num_cameras: int = 9,
        load_depth: bool = False,
    ):
        self.root_path = root_path
        if subset is None:
            subset = self.DEFAULT_SUBSET

        self.sample_list: List[str] = []
        for split in subset:
            split_dir = os.path.join(root_path, split)
            if not os.path.isdir(split_dir):
                continue
            names = sorted(os.listdir(split_dir))
            for name in names:
                self.sample_list.append(os.path.join(split_dir, name))

        self.is_static = False
        self._num_frames = num_frames
        self._num_cameras = num_cameras
        self.load_depth = load_depth

    def __len__(self):
        return len(self.sample_list)

    def count_frames(self, idx):
        return self._num_frames

    def count_cameras(self, idx):
        return self._num_cameras

    @staticmethod
    def _compute_c2w(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_wxyz_to_R(quat)
        T[:3, 3] = pos
        return T

    @staticmethod
    def _read_depth_tiff(buf: bytes) -> np.ndarray:
        """Decode a kubric ``depth_*.tiff`` to an ``(H, W)`` float32 array.

        Kubric stores per-pixel z-depth in meters as a single-channel float TIFF.
        """
        img = Image.open(BytesIO(buf))
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:  # safety: collapse trailing dim if any
            arr = arr[..., 0]
        return arr

    def get_data(
        self,
        idx: int,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
        camera_convention: str = "opencv",
        num_depth_frames: Optional[int] = None,
        **_ignored_kwargs,
    ):
        """Load RGB / camera / depth tensors for the given frames+views.

        Returns a dict in the shape expected by ``tokengs.data.provider``:
            DF_IMAGE_RGB           : (V*F, 3, H, W) float32 in [0, 1]
            DF_CAMERA_C2W_TRANSFORM: (V*F, 4, 4) float32, opencv convention
            DF_CAMERA_INTRINSICS   : (V*F, 4) float32, [fx, fy, cx, cy]
            DF_DEPTH (optional)    : (V*F, 1, H, W) float32. Per-view, depth
              is loaded for the first ``num_depth_frames`` frames and zeroed
              for the rest (matches DL3DV._load_depth_seq convention so that
              only the first ``num_input_views`` rows after curation
              contribute to the pointmap scale).

        Iteration order is view-major (all frames of view_indices[0],
        then all frames of view_indices[1], ...), matching the
        ``_curate_batch_dynamic`` reshape ``[v_in, v_in + 1]``.
        """
        assert camera_convention == "opencv"

        if frame_indices is None:
            frame_indices = list(range(self._num_frames))
        if view_indices is None:
            view_indices = list(range(self._num_cameras))

        sample_dir = self.sample_list[idx]

        want_rgb = DF_IMAGE_RGB in data_fields
        want_c2w = DF_CAMERA_C2W_TRANSFORM in data_fields
        want_intr = DF_CAMERA_INTRINSICS in data_fields
        want_depth = (DF_DEPTH in data_fields) and self.load_depth

        # Per-view, load depth for the first n_depth frames; zero otherwise.
        n_depth = (
            len(frame_indices)
            if num_depth_frames is None
            else min(num_depth_frames, len(frame_indices))
        )

        rgb_list: List[torch.Tensor] = []
        c2w_list: List[np.ndarray] = []
        intr_list: List[np.ndarray] = []
        depth_list: List[np.ndarray] = []

        for view_idx in view_indices:
            tar_path = os.path.join(sample_dir, f"output_{view_idx:03d}.tar")
            with tarfile.open(tar_path, "r") as tar:
                meta_buf = tar.extractfile("./metadata.json")
                assert meta_buf is not None, f"metadata.json missing in {tar_path}"
                meta = json.loads(meta_buf.read())

                focal = float(meta["camera"]["focal_length"])
                sensor_w = float(meta["camera"]["sensor_width"])
                H, W = meta["metadata"]["resolution"]
                fx = focal * W / sensor_w
                fy = focal * H / sensor_w
                cx = 0.5 * W
                cy = 0.5 * H
                intr = np.asarray([fx, fy, cx, cy], dtype=np.float32)

                positions = np.asarray(meta["camera"]["positions"], dtype=np.float64)
                quaternions = np.asarray(meta["camera"]["quaternions"], dtype=np.float64)

                for i_frame, frame_idx in enumerate(frame_indices):
                    if want_rgb:
                        img_buf = tar.extractfile(f"./rgba_{frame_idx:05d}.png")
                        assert img_buf is not None, (
                            f"rgba_{frame_idx:05d}.png missing in {tar_path}"
                        )
                        img = np.array(
                            Image.open(BytesIO(img_buf.read())).convert("RGB"),
                            dtype=np.uint8,
                        )
                        rgb_t = (
                            torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
                            / 255.0
                        )
                        rgb_list.append(rgb_t)

                    if want_intr:
                        intr_list.append(intr.copy())

                    if want_c2w:
                        c2w = self._compute_c2w(
                            positions[frame_idx], quaternions[frame_idx]
                        )
                        # kubric uses an opengl-style camera; flip y/z columns
                        # to match opencv (matches upstream btimer).
                        c2w[:3, 1:3] *= -1
                        c2w_list.append(c2w.astype(np.float32))

                    if want_depth:
                        if i_frame < n_depth:
                            depth_buf = tar.extractfile(
                                f"./depth_{frame_idx:05d}.tiff"
                            )
                            assert depth_buf is not None, (
                                f"depth_{frame_idx:05d}.tiff missing in {tar_path}"
                            )
                            depth = self._read_depth_tiff(depth_buf.read())
                        else:
                            depth = np.zeros((H, W), dtype=np.float32)
                        depth_list.append(depth)

        output = {"__key__": Path(sample_dir).stem}
        if want_rgb:
            output[DF_IMAGE_RGB] = torch.stack(rgb_list, dim=0)
        if want_c2w:
            output[DF_CAMERA_C2W_TRANSFORM] = torch.from_numpy(
                np.stack(c2w_list, axis=0)
            ).contiguous()
        if want_intr:
            output[DF_CAMERA_INTRINSICS] = torch.from_numpy(
                np.stack(intr_list, axis=0)
            ).contiguous()
        if want_depth:
            depth_arr = np.stack(depth_list, axis=0)  # (V*F, H, W)
            output[DF_DEPTH] = torch.from_numpy(depth_arr).float().unsqueeze(1).contiguous()

        return output
