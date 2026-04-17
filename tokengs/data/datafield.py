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

# Common data fields shared by most datasets
DF_IMAGE_RGB = "image_rgb"
# [B, 4, 4], float32, camera-to-world transformation matrix.
DF_CAMERA_C2W_TRANSFORM = "camera_c2w_transform"
# [B, 4], float32, pinhole intrinsics represented as [fx, fy, cx, cy].
DF_CAMERA_INTRINSICS = "camera_intrinsics"
# [B, H, W, 1], float32, foreground mask.
DF_FOREGROUND_MASK = "foreground_mask"
# [B, H, W, 1], float32, depth map.
DF_DEPTH = "depth"