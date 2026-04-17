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

from dataclasses import dataclass

import torch


@dataclass
class Gaussians:
    """Structured views over a single tensor of Gaussian parameters (14 channels)."""
    raw: torch.Tensor
    xyz: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    rgb: torch.Tensor

    @classmethod
    def from_raw(cls, raw: torch.Tensor) -> Gaussians:
        assert raw.shape[-1] == 14, f"Raw gaussians must have 14 channels, got {raw.shape[-1]}."
        return cls(
            raw=raw,
            xyz=raw[..., 0:3],
            opacity=raw[..., 3:4],
            scaling=raw[..., 4:7],
            rotation=raw[..., 7:11],
            rgb=raw[..., 11:14],
        )