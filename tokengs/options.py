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

"""Tyro CLI options and named presets (`AllConfigs` subcommands)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import tyro


@dataclass
class Options:
    # --- general
    evaluating: bool = False
    workspace: str = "./workspace"
    resume: str | None = None
    model_type: str = "tokengs"
    seed: int = 42

    # --- wandb / logging
    use_wandb: bool = False
    experiment_name: str = "tokengs"
    out_dir: str = "outputs"
    project_name: str = "TokenGS"

    # --- model architecture
    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 8
    dec_patch_size: int | None = None
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4
    clip_head_readout_std: float = 0.002
    clip_head_z_init: float | None = None
    dec_init_values: float | None = None

    # --- gaussian splatting
    bg_color: Literal["white", "black", "grey"] = "grey"
    gaussian_scale_cap: float = 0.075
    opacity_bias: float = 2.0
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    init_latents_from_existing: bool = False

    # --- dataset
    data_mode: tuple[tuple[str, int], ...] = (("dl3dv_scaled_0.15", 6),)
    num_views: int = 8
    num_input_views: int = 4
    znear: float = 0.025
    zfar: float = 125.0
    camera_normalization_method: Literal["mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound", "pointmap"] = "constant"
    pointmap_trim_lo: float = 0.0
    pointmap_trim_hi: float = 1.0
    num_workers: int = 16
    dataset_kwargs: dict[str, str] | None = None

    # --- training
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    max_iters_per_epoch: int = 1_000_000
    lr: float = 4e-4
    pct_start_steps: int = 1000
    final_div_factor: float = 1000.0
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    deferred_bp: bool = False
    use_input_supervision: bool = False
    mean_of_grads: Literal["none", "per-scene", "per-view"] = "none"
    mean_of_grads_scene_chunk_size: int = 1
    mean_of_grads_view_chunk_size: int | None = None

    # --- loss weights
    rgb_loss_type: Literal["l1", "l2"] = "l2"
    lambda_rgb: float = 1.0
    lambda_lpips: float = 0.0
    lambda_mask: float = 0.0
    lambda_ssim: float = 0.2
    lambda_visibility: float = 1.0
    visibility_distance_threshold: float = 1.0
    lambda_opacity: float = 0.0
    lambda_dyn_aux: float = 0.0
    lambda_dyn_aux_warmup_steps: int = 0
    lambda_dyn_aux_decay_steps: int = 0
    lambda_dyn_aux_min: float = 0.0

    # --- logging frequency
    print_freq: int = 10
    log_image_freq: int = 100

    # --- evaluation
    eval_n_media_dumps: int = 0
    strict_checkpoint_loading: bool = True

    # --- test-time training (eval)
    use_ttt_for_eval: bool = False
    ttt_mode: Literal["token-tuning", "scene-latent-tuning", "tokens", "latents"] = "token-tuning"
    ttt_n_steps: int = 50
    ttt_lr: float = 1e-4

    # --- dynamic scenes
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False

    # --- latent bottleneck release architecture
    use_multiscale_encoder: bool = False
    multiscale_encoder_layers: tuple[int, ...] = (5, 7, 9, 11)
    use_latent_bottleneck: bool = False
    num_latents: int = 4096
    latent_cross_attn_depth: int = 12

    # --- augmentation
    random_reflect: bool = True

    def __post_init__(self) -> None:
        if self.dec_patch_size is None:
            self.dec_patch_size = self.patch_size
        self.validate()

    def validate(self) -> None:
        if self.evaluating:
            assert not self.use_input_supervision, "use_input_supervision must be False when evaluating"
        if self.mean_of_grads not in ("none", "per-scene", "per-view"):
            raise ValueError("mean_of_grads must be one of: none, per-scene, per-view")
        if self.deferred_bp and self.mean_of_grads != "none":
            raise ValueError("deferred_bp and mean_of_grads are mutually exclusive backprop strategies")
        if self.mean_of_grads_scene_chunk_size <= 0:
            raise ValueError("mean_of_grads_scene_chunk_size must be positive")
        if self.mean_of_grads_view_chunk_size is not None and self.mean_of_grads_view_chunk_size <= 0:
            raise ValueError("mean_of_grads_view_chunk_size must be positive")

    def evolve(self, **changes: Any) -> Options:
        """Return a deep copy with the given fields replaced."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        new_instance.validate()
        return new_instance


config_defaults: dict[str, Options] = {}
config_doc: dict[str, str] = {}

config_doc["train_dl3dv_base"] = "DL3DV training defaults (long schedule, capped iters/epoch)."
config_defaults["train_dl3dv_base"] = Options(
    num_epochs=300,
    max_iters_per_epoch=500,
    pct_start_steps=2000,
)

config_doc["finetune_dl3dv_2view"] = "Short finetune from existing tokens, 2 input views, wide images."
config_defaults["finetune_dl3dv_2view"] = config_defaults["train_dl3dv_base"].evolve(
    num_epochs=20,
    pct_start_steps=400,
    lr=4e-5,
    num_gs_tokens=4096,
    init_tokens_from_existing=True,
    num_input_views=2,
    img_size=(256, 448),
)

config_doc["finetune_dl3dv_4view"] = "Like finetune_dl3dv_2view with 4 input views."
config_defaults["finetune_dl3dv_4view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=4,
)

config_doc["finetune_dl3dv_6view"] = "Like finetune_dl3dv_2view with 6 input views and 10 total views."
config_defaults["finetune_dl3dv_6view"] = config_defaults["finetune_dl3dv_2view"].evolve(
    num_input_views=6,
    num_views=10,
)

config_doc["eval_dl3dv_2view"] = "DL3DV eval preset: 2 views, eval JSON, single batch."
config_defaults["eval_dl3dv_2view"] = Options(
    data_mode=(("dl3dv_eval_scaled_0.15", 1),),
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_2v.json"},
    num_input_views=2,
    img_size=(256, 448),
    evaluating=True,
    num_gs_tokens=4096,
    use_input_supervision=False,
    batch_size=1,
)

config_doc["eval_dl3dv_4view"] = "DL3DV eval preset: 4 input views."
config_defaults["eval_dl3dv_4view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=4,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_4v.json"},
)
config_doc["eval_dl3dv_6view"] = "DL3DV eval preset: 6 input views."
config_defaults["eval_dl3dv_6view"] = config_defaults["eval_dl3dv_2view"].evolve(
    num_input_views=6,
    dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_6v.json"},
)


# ----- DL3DV latent-bottleneck release presets -----
_LATENT_D12_ARCH = {
    "enc_depth": 12,
    "dec_depth": 1,
    "dec_patch_size": 8,
    "clip_head_readout_std": 0.002,
    "clip_head_z_init": 0.1,
    "dec_init_values": 0.01,
    "gaussian_z_offset": 0.0,
    "opacity_bias": 2.0,
    "gs_token_std": 0.02,
    "use_multiscale_encoder": True,
    "multiscale_encoder_layers": (5, 7, 9, 11),
    "use_latent_bottleneck": True,
    "num_latents": 4096,
    "latent_cross_attn_depth": 12,
    "camera_normalization_method": "mean_cam",
    "camera_scale_method": "constant",
}

_SSIM_LOSS = {
    "rgb_loss_type": "l1",
    "lambda_rgb": 0.8,
    "lambda_ssim": 0.2,
    "lambda_lpips": 0.0,
}

_LPIPS_LOSS = {
    "rgb_loss_type": "l2",
    "lambda_rgb": 1.0,
    "lambda_ssim": 0.0,
    "lambda_lpips": 0.5,
}


def _latent_dl3dv_train_preset(num_input_views: int, num_views: int) -> Options:
    return config_defaults["train_dl3dv_base"].evolve(
        data_mode=(("dl3dv_scaled_0.15", 6),),
        num_epochs=20,
        pct_start_steps=400,
        lr=4e-5,
        num_gs_tokens=4096,
        init_tokens_from_existing=True,
        init_latents_from_existing=True,
        num_input_views=num_input_views,
        num_views=num_views,
        img_size=(256, 448),
        **_LATENT_D12_ARCH,
    )


def _latent_dl3dv_eval_preset(num_input_views: int, evaluation_json: str) -> Options:
    return Options(
        data_mode=(("dl3dv_eval_scaled_0.15", 1),),
        dataset_kwargs={"evaluation_json": evaluation_json},
        num_input_views=num_input_views,
        img_size=(256, 448),
        evaluating=True,
        num_gs_tokens=4096,
        use_input_supervision=False,
        ttt_mode="scene-latent-tuning",
        ttt_lr=1e-2,
        batch_size=1,
        **_LATENT_D12_ARCH,
    )


for _num_input_views, _num_views in ((2, 8), (4, 8), (6, 10)):
    _name = f"finetune_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_SSIM_LOSS)

    _name = f"finetune_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Finetune the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS model."
    )
    config_defaults[_name] = _latent_dl3dv_train_preset(
        _num_input_views, _num_views
    ).evolve(**_LPIPS_LOSS)

for _num_input_views in (2, 4, 6):
    _eval_json = f"assets/evaluation_idx_dl3dv_depthsplat_{_num_input_views}v.json"

    _name = f"eval_dl3dv_latent_{_num_input_views}view_ssim"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view SSIM checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_SSIM_LOSS)

    _name = f"eval_dl3dv_latent_{_num_input_views}view_lpips"
    config_doc[_name] = (
        f"Evaluate the released SA d12 latent-bottleneck DL3DV {_num_input_views}-view LPIPS checkpoint."
    )
    config_defaults[_name] = _latent_dl3dv_eval_preset(
        _num_input_views, _eval_json
    ).evolve(**_LPIPS_LOSS)


# ----- Kubric 4D dynamic finetune -----
config_doc["finetune_dl3dv_kubric_dyn"] = (
    "Dynamic finetune of the DL3DV base on Kubric4D. Adds dynamic GS tokens "
    "warm-started from the static tokens, sinusoidal time embeddings, "
    "interpolated target-frame sampling, and pointmap camera scaling."
)
config_defaults["finetune_dl3dv_kubric_dyn"] = config_defaults["train_dl3dv_base"].evolve(
    use_input_supervision=False,
    data_mode=(("kubric_scaled_1.0", 100),),
    dataset_kwargs={},
    num_input_views=4,
    num_views=8,
    img_size=(256, 256),
    num_epochs=100,
    max_iters_per_epoch=500,
    pct_start_steps=2500,
    lr=4e-5,
    init_tokens_from_existing=True,
    num_gs_tokens=1024,
    num_dynamic_gs_tokens=256,
    init_dynamic_tokens_from_static=True,
    time_embedding=True,
    time_embedding_dim=2,
    use_interp_target=True,
    camera_normalization_method="mean_cam",
    camera_scale_method="pointmap",
    enc_depth=12,
    dec_depth=1,
    dec_patch_size=8,
    clip_head_readout_std=0.002,
    clip_head_z_init=0.1,
    dec_init_values=0.01,
    gaussian_z_offset=0.0,
    opacity_bias=2.0,
    gs_token_std=0.02,
    use_multiscale_encoder=True,
    multiscale_encoder_layers=(5, 7, 9, 11),
    use_latent_bottleneck=True,
    num_latents=4096,
    latent_cross_attn_depth=12,
    rgb_loss_type="l1",
    lambda_rgb=0.8,
    lambda_ssim=0.2,
    lambda_visibility=1.0,
    lambda_opacity=0.0,
    project_name="TokenGS-Kubric",
)

config_doc["finetune_dl3dv_kubric_static"] = (
    "Stage-1 Kubric domain finetune without dynamic tokens or time embeddings. "
    "Use this when reproducing the two-stage warm-start variant."
)
config_defaults["finetune_dl3dv_kubric_static"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    num_dynamic_gs_tokens=0,
    init_dynamic_tokens_from_static=False,
    time_embedding=False,
)

config_doc["finetune_dl3dv_kubric_dyn_v2"] = (
    "Kubric dynamic finetune with an auxiliary dynamic-only render loss."
)
config_defaults["finetune_dl3dv_kubric_dyn_v2"] = config_defaults["finetune_dl3dv_kubric_dyn"].evolve(
    lambda_dyn_aux=0.3,
)

_kubric_dyn_release = config_defaults["finetune_dl3dv_kubric_dyn_v2"].evolve(
    lambda_dyn_aux=0.3,
    lambda_dyn_aux_warmup_steps=5000,
    lambda_dyn_aux_decay_steps=10000,
    lambda_dyn_aux_min=0.0,
)
config_doc["finetune_dl3dv_kubric_dyn_release"] = (
    "Released Kubric dynamic finetune schedule: hold lambda_dyn_aux=0.3 for "
    "5K steps, then linearly decay it to 0 over 10K steps."
)
config_defaults["finetune_dl3dv_kubric_dyn_release"] = _kubric_dyn_release
config_doc["finetune_dl3dv_kubric_dyn_v3"] = (
    "Backward-compatible alias for finetune_dl3dv_kubric_dyn_release."
)
config_defaults["finetune_dl3dv_kubric_dyn_v3"] = _kubric_dyn_release

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
