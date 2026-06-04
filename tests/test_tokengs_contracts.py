import unittest

import torch
from gsplat.rendering import rasterization

from tokengs.models.input_types import ModelInputDecoder, Reconstruction, split_data
from tokengs.models.losses import project_gaussian_means2d
from tokengs.models.tokengs import TokenGS
from tokengs.options import Options


class FakeRenderer:
    def __init__(self, img_size):
        self.img_size = img_size
        self.calls = []

    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None):
        self.calls.append(
            {
                "gaussians": gaussians,
                "cam_view": cam_view,
                "bg_color": bg_color,
                "intrinsics": intrinsics,
            }
        )
        B, V = cam_view.shape[:2]
        N = gaussians.shape[1]
        H, W = self.img_size
        base = gaussians[..., 0].mean(dim=1).view(B, 1, 1, 1, 1)
        images = torch.zeros(B, V, 3, H, W, dtype=gaussians.dtype, device=gaussians.device) + base
        alphas = torch.ones(B, V, 1, H, W, dtype=gaussians.dtype, device=gaussians.device) * torch.sigmoid(base)
        depths = torch.ones(B, V, 1, H, W, dtype=gaussians.dtype, device=gaussians.device)
        means2d = torch.zeros(B, V, N, 2, dtype=gaussians.dtype, device=gaussians.device)
        return {
            "images_pred": images,
            "alphas_pred": alphas,
            "depths_pred": depths,
            "means2d_pred": means2d,
        }


class FakeLPIPSLoss(torch.nn.Module):
    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if in0.ndim != 4 or in1.ndim != 4:
            raise AssertionError("LPIPS inputs should be flattened to [B, C, H, W]")
        if normalize is not True:
            raise AssertionError("LPIPS should be called with normalize=True")
        return (in0 - in1).square().mean(dim=(1, 2, 3), keepdim=True)


def make_tiny_model(**overrides) -> TokenGS:
    option_kwargs = {
        "img_size": (8, 8),
        "patch_size": 4,
        "enc_depth": 1,
        "dec_depth": 1,
        "enc_embed_dim": 8,
        "enc_num_heads": 2,
        "mlp_ratio": 1,
        "num_gs_tokens": 2,
        "token_dim": 8,
        "num_input_views": 1,
        "num_views": 2,
        "lambda_lpips": 0.0,
        "lambda_mask": 0.0,
        "lambda_ssim": 0.0,
        "lambda_visibility": 0.0,
        "random_reflect": False,
    }
    option_kwargs.update(overrides)
    opt = Options(
        **option_kwargs,
    )
    model = TokenGS(opt)
    model.gs = FakeRenderer(opt.img_size)
    return model


def make_batch(batch_size: int = 1, target_views: int = 1) -> dict:
    H, W = 8, 8
    num_views = 2
    num_input_views = 1
    dtype = torch.float32

    input_tensor = torch.zeros(batch_size, num_views, 9, H, W, dtype=dtype)
    intrinsics = torch.tensor([4.0, 4.0, W / 2, H / 2], dtype=dtype).view(1, 1, 4)
    intrinsics = intrinsics.repeat(batch_size, target_views, 1)
    cam_view = torch.eye(4, dtype=dtype).view(1, 1, 4, 4).repeat(batch_size, target_views, 1, 1)
    cam_to_world_input = torch.eye(4, dtype=dtype).view(1, 1, 4, 4).repeat(batch_size, num_input_views, 1, 1)

    return {
        "input": input_tensor,
        "rays_os": torch.zeros(batch_size, num_views, 3, H, W, dtype=dtype),
        "rays_ds": torch.zeros(batch_size, num_views, 3, H, W, dtype=dtype),
        "images_input": torch.zeros(batch_size, num_input_views, 3, H, W, dtype=dtype),
        "images_output": torch.zeros(batch_size, target_views, 3, H, W, dtype=dtype),
        "intrinsics": intrinsics,
        "intrinsics_input": intrinsics[:, :num_input_views],
        "cam_view": cam_view,
        "masks_output": torch.ones(batch_size, target_views, 1, H, W, dtype=dtype),
        "has_mask": torch.zeros(batch_size, dtype=torch.bool),
        "cam_to_world_input": cam_to_world_input,
    }


class TestOptionsValidation(unittest.TestCase):
    def test_deferred_bp_and_mean_of_grads_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "deferred_bp and mean_of_grads"):
            Options(deferred_bp=True, mean_of_grads="per-scene")
        with self.assertRaisesRegex(ValueError, "deferred_bp and mean_of_grads"):
            Options(deferred_bp=True, mean_of_grads="per-view")

    def test_mean_of_grads_validates_mode(self):
        with self.assertRaisesRegex(ValueError, "mean_of_grads"):
            Options(mean_of_grads="scene")
        with self.assertRaisesRegex(ValueError, "mean_of_grads_scene_chunk_size"):
            Options(mean_of_grads_scene_chunk_size=0)
        with self.assertRaisesRegex(ValueError, "mean_of_grads_view_chunk_size"):
            Options(mean_of_grads_view_chunk_size=0)

    def test_evolve_validates_deferred_bp_and_mean_of_grads(self):
        opt = Options(deferred_bp=True)

        with self.assertRaisesRegex(ValueError, "deferred_bp and mean_of_grads"):
            opt.evolve(mean_of_grads="per-view")


class TestTokenGSContracts(unittest.TestCase):
    def test_select_batch_preserves_batch_dimension_and_slices_views(self):
        model = make_tiny_model()
        model_input, supervision = split_data(make_batch(batch_size=2, target_views=3), model.opt)

        decoder_b = model_input.decoder.select_batch(1, slice(1, 3))
        supervision_b = supervision.select_batch(1, slice(1, 3))

        self.assertEqual(decoder_b.cam_view.shape[0], 1)
        self.assertEqual(decoder_b.cam_view.shape[1], 2)
        self.assertEqual(decoder_b.intrinsics.shape[0], 1)
        self.assertEqual(decoder_b.intrinsics.shape[1], 2)
        self.assertEqual(supervision_b.images_output.shape[0], 1)
        self.assertEqual(supervision_b.images_output.shape[1], 2)
        self.assertEqual(supervision_b.has_mask.shape[0], 1)

    def test_forward_reconstruction_contract(self):
        model = make_tiny_model()
        model_input, _ = split_data(make_batch(), model.opt)

        reconstruction = model.forward_reconstruction(model_input)

        self.assertIsInstance(reconstruction, Reconstruction)
        self.assertEqual(reconstruction.gaussians.shape[-1], 14)
        self.assertEqual(reconstruction.background_color.shape, (3,))
        self.assertEqual(reconstruction.background_color.dtype, reconstruction.gaussians.dtype)
        self.assertEqual(reconstruction.background_color.device, reconstruction.gaussians.device)

    def test_render_reconstruction_matches_render_gaussians_contract(self):
        model = make_tiny_model()
        gaussians = torch.randn(1, 3, 14)
        decoder_input = ModelInputDecoder(
            cam_view=torch.eye(4).view(1, 1, 4, 4),
            intrinsics=torch.tensor([[[4.0, 4.0, 4.0, 4.0]]]),
        )
        reconstruction = Reconstruction(
            gaussians=gaussians,
            background_color=model._background_color(gaussians.dtype, gaussians.device),
        )

        from_reconstruction = model.render_reconstruction(reconstruction, decoder_input)
        from_gaussians = model.render_gaussians(gaussians, decoder_input.cam_view, decoder_input.intrinsics)

        self.assertEqual(set(from_reconstruction.keys()), set(from_gaussians.keys()))
        for key in from_reconstruction:
            torch.testing.assert_close(from_reconstruction[key], from_gaussians[key])

    def test_forward_result_contract_and_skip_loss(self):
        model = make_tiny_model()
        batch = make_batch()

        out = model(batch)

        for key in ("gaussians", "loss", "loss_rgb", "psnr", "images_pred", "alphas_pred", "depths_pred", "images_output"):
            self.assertIn(key, out)

        skip_out = model(batch, skip_loss=True)

        for key in ("gaussians", "images_pred", "alphas_pred", "depths_pred", "means2d_pred"):
            self.assertIn(key, skip_out)
        self.assertNotIn("loss", skip_out)

    def _assert_mean_of_grads_matches_metrics_and_gradients(
        self,
        mode: str,
        scene_chunk_size: int = 1,
        view_chunk_size: int | None = None,
        lambda_lpips: float = 0.0,
    ):
        torch.manual_seed(0)
        model = make_tiny_model(lambda_mask=0.7, lambda_visibility=0.5, visibility_distance_threshold=0.0)
        if lambda_lpips > 0:
            model.opt.lambda_lpips = lambda_lpips
            model.lpips_loss = FakeLPIPSLoss()
        batch = make_batch(batch_size=2, target_views=2)
        batch["has_mask"] = torch.tensor([True, False])
        batch["intrinsics"] = torch.tensor([20.0, 20.0, 0.0, 0.0]).view(1, 1, 4).repeat(2, 2, 1)

        model.opt.mean_of_grads = "none"
        out_normal = model(batch)
        out_normal["loss"].backward()
        normal_grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        model.zero_grad(set_to_none=True)
        model.gs.calls.clear()
        model.opt.mean_of_grads = mode
        model.opt.mean_of_grads_scene_chunk_size = scene_chunk_size
        model.opt.mean_of_grads_view_chunk_size = view_chunk_size
        out_mean = model(batch)
        out_mean["backward_loss"].backward()
        mean_grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        self.assertIn("backward_loss", out_mean)
        scalar_keys = ["loss", "loss_rgb", "loss_mask", "loss_visibility", "psnr"]
        per_scene_keys = ["loss_per_scene", "loss_rgb_per_scene", "loss_mask_per_scene", "loss_visibility_per_scene", "psnr_per_scene"]
        if lambda_lpips > 0:
            scalar_keys.append("loss_lpips")
            per_scene_keys.append("loss_lpips_per_scene")
        for key in scalar_keys:
            torch.testing.assert_close(out_mean[key], out_normal[key].detach(), atol=1e-6, rtol=1e-5)
        for key in per_scene_keys:
            torch.testing.assert_close(out_mean[key], out_normal[key].detach(), atol=1e-6, rtol=1e-5)
        for key in ("images_pred", "alphas_pred", "depths_pred"):
            self.assertEqual(out_mean[key].shape, out_normal[key].shape)

        batch_size, num_views = batch["images_output"].shape[:2]
        effective_view_chunk_size = view_chunk_size if view_chunk_size is not None else (num_views if mode == "per-scene" else 1)
        expected_calls = (
            (batch_size + scene_chunk_size - 1) // scene_chunk_size
            * ((num_views + effective_view_chunk_size - 1) // effective_view_chunk_size)
        )
        self.assertEqual(len(model.gs.calls), expected_calls)
        self.assertTrue(all(call["cam_view"].shape[1] <= effective_view_chunk_size for call in model.gs.calls))
        self.assertTrue(all(call["gaussians"].shape[0] <= scene_chunk_size for call in model.gs.calls))

        self.assertEqual(set(mean_grads.keys()), set(normal_grads.keys()))
        for name in normal_grads:
            torch.testing.assert_close(mean_grads[name], normal_grads[name], atol=1e-6, rtol=1e-5)

    def test_mean_of_grads_per_scene_matches_metrics_and_gradients(self):
        self._assert_mean_of_grads_matches_metrics_and_gradients("per-scene")

    def test_mean_of_grads_per_view_matches_metrics_and_gradients(self):
        self._assert_mean_of_grads_matches_metrics_and_gradients("per-view")

    def test_mean_of_grads_custom_chunks_match_metrics_and_gradients(self):
        self._assert_mean_of_grads_matches_metrics_and_gradients(
            "per-view",
            scene_chunk_size=2,
            view_chunk_size=1,
        )

    def test_mean_of_grads_with_lpips_matches_metrics_and_gradients(self):
        self._assert_mean_of_grads_matches_metrics_and_gradients(
            "per-view",
            scene_chunk_size=1,
            view_chunk_size=1,
            lambda_lpips=0.3,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for gsplat projection comparison")
    def test_project_gaussian_means2d_matches_gsplat_for_rendered_gaussians(self):
        device = torch.device("cuda")
        means = torch.tensor(
            [[[0.0, 0.0, 2.0], [1.0, 0.25, 2.0], [-0.5, 0.5, 3.0]]],
            device=device,
            requires_grad=True,
        )
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 3], device=device)
        scales = torch.ones(1, 3, 3, device=device) * 0.5
        opacities = torch.ones(1, 3, device=device)
        colors = torch.ones(1, 3, 3, device=device)
        viewmats = torch.eye(4, device=device).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[4.0, 4.0, 4.0, 4.0]]], device=device)
        Ks = torch.tensor([[[[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]]], device=device)

        _, _, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=8,
            height=8,
            packed=False,
            render_mode="RGB",
        )

        projected = project_gaussian_means2d(means, viewmats.transpose(-1, -2), intrinsics)
        rendered = info["means2d"]
        rendered_mask = (info["radii"][..., 0] > 0) & (info["radii"][..., 1] > 0)
        torch.testing.assert_close(projected[rendered_mask], rendered[rendered_mask], atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for gsplat projection comparison")
    def test_project_gaussian_means2d_regularizes_culled_gaussians(self):
        device = torch.device("cuda")
        means = torch.tensor([[[8.0, 0.0, 2.0]]], device=device, requires_grad=True)
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)
        scales = torch.ones(1, 1, 3, device=device) * 0.05
        opacities = torch.ones(1, 1, device=device)
        colors = torch.ones(1, 1, 3, device=device)
        viewmats = torch.eye(4, device=device).view(1, 1, 4, 4)
        intrinsics = torch.tensor([[[4.0, 4.0, 4.0, 4.0]]], device=device)
        Ks = torch.tensor([[[[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]]], device=device)

        _, _, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=8,
            height=8,
            packed=False,
            render_mode="RGB",
        )

        projected = project_gaussian_means2d(means, viewmats.transpose(-1, -2), intrinsics)
        self.assertEqual(int(info["radii"][0, 0, 0, 0]), 0)
        torch.testing.assert_close(projected.detach(), torch.tensor([[[[20.0, 4.0]]]], device=device))

        torch.nan_to_num(info["means2d"], nan=0.0, posinf=1e6, neginf=-1e6).sum().backward(retain_graph=True)
        torch.testing.assert_close(means.grad, torch.zeros_like(means.grad))
        means.grad.zero_()

        projected[..., 0].sum().backward()
        self.assertNotEqual(float(means.grad[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
