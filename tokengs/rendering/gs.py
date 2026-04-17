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
from gsplat.rendering import rasterization


class DeferredBP(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_w2c, test_intr, 
               W, H, near_plane, far_plane, backgrounds):
        rgbd, alpha, info = rasterization(
            means=xyz, 
            quats=rotation, 
            scales=scale, 
            opacities=opacity, 
            colors=feature,
            viewmats=test_w2c, 
            Ks=test_intr, 
            width=W, 
            height=H, 
            near_plane=near_plane, 
            far_plane=far_plane,
            backgrounds=backgrounds,
            render_mode="RGB+ED", 
            packed=False,   # important for correct means2d shape
        ) # (1, H, W, 3) 
        image, depth = rgbd[..., :3], rgbd[..., 3:]
        return image, alpha, depth, info['means2d']     # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_w2cs, test_intr,
                W, H, near_plane, far_plane, backgrounds):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds)
        ctx.W = W
        ctx.H = H
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        N = xyz.shape[1]
        with torch.no_grad():
            B, V = test_intr.shape[:2]
            images = torch.zeros(B, V, H, W, 3).to(xyz.device)
            alphas = torch.zeros(B, V, H, W, 1).to(xyz.device)
            depths = torch.zeros(B, V, H, W, 1).to(xyz.device)
            means2ds = torch.zeros(B, V, N, 2).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth, means2d = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1]
                    )
                    images[ib, iv:iv+1] = image
                    alphas[ib, iv:iv+1] = alpha
                    depths[ib, iv:iv+1] = depth
                    means2ds[ib, iv:iv+1] = means2d
        images = images.requires_grad_()
        alphas = alphas.requires_grad_()
        depths = depths.requires_grad_()
        means2ds = means2ds.requires_grad_()
        return images, alphas, depths, means2ds

    @staticmethod
    def backward(ctx, images_grad, alphas_grad, depths_grad, means2ds_grad):
        xyz, feature, scale, rotation, opacity, test_w2cs, test_intr, backgrounds = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V = test_intr.shape[:2]
            for ib in range(B):
                for iv in range(V):
                    image, alpha, depth, means2d = DeferredBP.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                        test_w2cs[ib,iv:iv+1], test_intr[ib,iv:iv+1], 
                        W, H, near_plane, far_plane, backgrounds[ib,iv:iv+1]
                    )
                    render_split = torch.cat([image.reshape(-1), alpha.reshape(-1), depth.reshape(-1), means2d.reshape(-1)], dim=-1)
                    grad_split = torch.cat([images_grad[ib, iv:iv+1].reshape(-1), alphas_grad[ib, iv:iv+1].reshape(-1), depths_grad[ib, iv:iv+1].reshape(-1), means2ds_grad[ib, iv:iv+1].reshape(-1)], dim=-1) 
                    render_split.backward(grad_split)

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None

class GaussianRenderer:
    def __init__(self, opt):
        self.opt = opt
        
    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None):
        B, V = cam_view.shape[:2]
        # pos, opacity, scale, rotation, shs
        means3D = gaussians[..., 0:3].contiguous().float()
        opacity = gaussians[..., 3:4].contiguous().float().squeeze(-1)
        scales = gaussians[..., 4:7].contiguous().float()
        rotations = gaussians[..., 7:11].contiguous().float()
        rgbs = gaussians[..., 11:].contiguous().float() # [N, 3]

        viewmat = cam_view.float().transpose(3, 2)  # [B, V, 4, 4]
        Ks = torch.tensor([[[[view_intrinsic[0],0.,view_intrinsic[2]],[0.,view_intrinsic[1],view_intrinsic[3]],[0., 0., 1.]] for view_intrinsic in batch_intrinsic] for batch_intrinsic in intrinsics], dtype=means3D.dtype, device=means3D.device)
        backgrounds = bg_color[None, None].repeat(B, V, 1).to(means3D.device, means3D.dtype) if bg_color is not None else torch.ones(B, V, 3, dtype=means3D.dtype, device=means3D.device)

        H, W = self.opt.img_size
        near_plane, far_plane = self.opt.znear, self.opt.zfar
            
        if self.opt.deferred_bp:
            return self.render_deferred(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane)
        else:
            return self.render_standard(means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane)


    def render_deferred(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane):
        images, alphas, depths, means2ds = DeferredBP.apply(means3D, rgbs, scales, rotations, opacity, viewmat, Ks, W, H, near_plane, far_plane, backgrounds)
        return {
            "images_pred": images.permute(0,1,4,2,3), # [B, V, 3, H, W]
            "alphas_pred": alphas.permute(0,1,4,2,3), # [B, V, 1, H, W]
            "depths_pred": depths.permute(0,1,4,2,3), # [B, V, 1, H, W]
            "means2d_pred": means2ds, # [B, V, N, 2]
        }
                
                
    def render_standard(self, means3D, opacity, scales, rotations, rgbs, viewmat, Ks, backgrounds, H, W, near_plane, far_plane):
        # gaussians: [B, N, 14]
        # cam_pos: [B, V, 3]
        B, V = Ks.shape[:2]

        # loop of loop...
        images, alphas, depths, means2ds = [], [], [], []
        for b in range(B):
            rendered_image_all, rendered_alpha_all, info = rasterization(
                means=means3D[b],
                quats=rotations[b],
                scales=scales[b],
                opacities=opacity[b],
                colors=rgbs[b],
                viewmats=viewmat[b],
                Ks=Ks[b],
                width=W,
                height=H,
                near_plane=near_plane,
                far_plane=far_plane,
                packed=False,
                backgrounds=backgrounds[b],
                render_mode="RGB+ED",
            )
            for rendered_image, rendered_alpha, means2d in zip(rendered_image_all, rendered_alpha_all, info['means2d']):
                depths.append(rendered_image[...,3:].permute(2, 0, 1))
                rendered_image = rendered_image[...,:3].permute(2, 0, 1)
                images.append(rendered_image)
                alphas.append(rendered_alpha.permute(2, 0, 1))
                means2ds.append(means2d) # [N, 2]
        images, alphas, depths, means2ds = torch.stack(images), torch.stack(alphas), torch.stack(depths), torch.stack(means2ds)
        images, alphas, depths, means2ds = images.view(B, V, *images.shape[1:]), alphas.view(B, V, *alphas.shape[1:]), depths.view(B, V, *depths.shape[1:]), means2ds.view(B, V, *means2ds.shape[1:])

        return {
            "images_pred": images, # [B, V, 3, H, W]
            "alphas_pred": alphas, # [B, V, 1, H, W]
            "depths_pred": depths, # [B, V, 1, H, W]
            "means2d_pred": means2ds, # [B, V, N, 2]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = torch.logit(opacity.clamp(1e-8, 1 - 1e-8))  # inverse sigmoid
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)