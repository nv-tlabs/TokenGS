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

"""
Original fused_ssim doesn't support vmap. This version provides re-wraps their CUDA ops to support it.
"""

import torch

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward
elif torch.mps.is_available():
    from fused_ssim_mps import fusedssim, fusedssim_backward
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    from fused_ssim_xpu import fusedssim, fusedssim_backward


allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):

    @staticmethod
    def forward(C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        # Use view_as to avoid "input returned as output" error with setup_context
        return ssim_map, img1.view_as(img1), img2.view_as(img2), dm_dmu1, dm_dsigma1_sq, dm_dsigma12, C1, C2, padding

    @staticmethod
    def setup_context(ctx, inputs, output):
        C1, C2, img1, img2, padding, train = inputs
        ssim_map, img1_out, img2_out, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, C1_out, C2_out, padding_out = output
        
        ctx.save_for_backward(img1_out.detach(), img2_out, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1_out
        ctx.C2 = C2_out
        ctx.padding = padding_out
        
        # Mark all outputs except ssim_map as non-differentiable
        ctx.mark_non_differentiable(img1_out, img2_out, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)

    @staticmethod
    def backward(ctx, grad_ssim_map, *args):
        # *args will capture gradients for the extra outputs (img1, img2, etc.)
        # We only care about grad_ssim_map
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        
        dL_dmap = grad_ssim_map
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = grad_ssim_map
        
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        # Return gradients: C1, C2, img1, img2, padding, train
        # Only img1 gets a gradient, rest are None
        return None, None, grad, None, None, None

    @staticmethod
    def vmap(info, in_dims, C1, C2, img1, img2, padding="same", train=True):
        # in_dims: (C1_dim, C2_dim, img1_dim, img2_dim, padding_dim, train_dim)
        # C1, C2, padding, and train are not tensors, so their dims should be None
        C1_dim, C2_dim, img1_dim, img2_dim, padding_dim, train_dim = in_dims
        
        # Determine which dimension is being vmapped and get the vmap size
        vmap_dim = img1_dim if img1_dim is not None else img2_dim
        
        if vmap_dim is None:
            # No vmapping happening, just call forward normally
            result_tuple = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
            # Return tuple with None out_dims for all elements
            return result_tuple, (None,) * len(result_tuple)
        
        # Move vmap dimension to front and fold into batch dimension
        # Handles nested vmap: [V1, V2, ..., B, C, H, W] -> flatten all leading dims into batch
        
        original_batch_size = None  # Track if we merged a batch dimension
        
        if img1_dim is not None:
            img1 = img1.movedim(img1_dim, 0)
            vmap_size = img1.shape[0]
            # Get shape after vmap dim: if [V, B, C, H, W] this is [B, C, H, W]
            # Flatten vmap dim into batch: [V, B, C, H, W] -> [V*B, C, H, W]
            # This works for nested vmap too: [V, C, H, W] -> [V, C, H, W] (already correct)
            if img1.ndim > 4:  # Has existing batch dimension to merge with
                original_batch_size = img1.shape[1]
                img1 = img1.reshape(vmap_size * original_batch_size, *img1.shape[2:])
            # else: already 4D [V, C, H, W], vmap_size IS the batch dimension
        
        if img2_dim is not None:
            img2 = img2.movedim(img2_dim, 0)
            vmap_size = img2.shape[0]
            if img2.ndim > 4:  # Has existing batch dimension to merge with
                if original_batch_size is None:
                    original_batch_size = img2.shape[1]
                img2 = img2.reshape(vmap_size * original_batch_size, *img2.shape[2:])
            # else: already 4D [V, C, H, W], vmap_size IS the batch dimension
        
        # Apply the function with merged batch dimension
        # This returns a tuple: (ssim_map, img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, C1, C2, padding)
        result_tuple = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
        
        # Unfold tensor outputs: split the merged batch dimension back out
        # If input was [V, B, C, H, W] -> [V*B, C, H, W] -> [V, B, C, H, W]
        # If input was [V, C, H, W] -> [V, C, H, W] (no change needed)
        def unfold_tensor(tensor, vmap_size, original_batch_size):
            if original_batch_size is not None:
                # Batch dimension was merged, split it back: [V*B, ...] -> [V, B, ...]
                return tensor.reshape(vmap_size, original_batch_size, *tensor.shape[1:])
            else:
                # No batch dimension was merged, tensor is already [V, ...], just return it
                return tensor
        
        ssim_map = unfold_tensor(result_tuple[0], vmap_size, original_batch_size)
        img1_out = unfold_tensor(result_tuple[1], vmap_size, original_batch_size)
        img2_out = unfold_tensor(result_tuple[2], vmap_size, original_batch_size)
        dm_dmu1 = unfold_tensor(result_tuple[3], vmap_size, original_batch_size)
        dm_dsigma1_sq = unfold_tensor(result_tuple[4], vmap_size, original_batch_size)
        dm_dsigma12 = unfold_tensor(result_tuple[5], vmap_size, original_batch_size)
        
        # Non-tensor outputs (C1, C2, padding) stay as is
        unfolded_result = (ssim_map, img1_out, img2_out, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, 
                          result_tuple[6], result_tuple[7], result_tuple[8])
        
        # The vmap dimension is at position 0 for tensor outputs, None for non-tensor outputs
        out_dims = (0, 0, 0, 0, 0, 0, None, None, None)
        
        return unfolded_result, out_dims

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    # FusedSSIMMap.apply returns a tuple, but we only need the first element (ssim_map)
    # The other elements are marked as non-differentiable so they don't interfere with gradients
    result = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    ssim_map = result[0] if isinstance(result, tuple) else result
    return ssim_map.mean()