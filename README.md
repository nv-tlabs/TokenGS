# TokenGS

![Teaser: TokenGS results and exploration](assets/tokengs_explore.gif)

**TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens** <br>
Jiawei Ren*, Michal Tyszkiewicz*, Jiahui Huang†, Zan Gojcic† <br>
\* indicates equal contribution, † indicates equal advising

[**Paper**](https://arxiv.org/abs/2604.15239) · [**Project Page**](https://research.nvidia.com/labs/toronto-ai/tokengs/) · [**HuggingFace**](https://huggingface.co/jiaweir/tokengs)

TokenGS predicts 3D Gaussians with a self-supervised rendering objective. An encoder–decoder stacks learnable Gaussian tokens so the number of primitives is not tied to image resolution or view count.

## News

- **2026.6.3:** Model improvement release: Latent bottleneck models, Scene-latent tuning, and Mean-of-gradients training. See the [release_2026.6](release_2026.6.md) note.
- **2026.6.3:** Released Kubric training and inference.
- **2026.6.2:** Released TokenGS model weights on [HuggingFace](https://huggingface.co/jiaweir/tokengs).

## Installation

Install the package in editable mode (dependencies include PyTorch, gsplat, and [fused-ssim](https://github.com/rahul-goel/fused-ssim) via `pyproject.toml`):

```bash
uv pip install -e .
```

**Environment:** Python 3.11, CUDA 12.6+ (see `pyproject.toml` for pinned versions).

**Data:** DL3DV layout, symlinks, and `dataset_kwargs` are described in **[data/DATA.md](data/DATA.md)**.

## Evaluation

Place weights under `checkpoints/` (or pass any path to `--resume`). Metrics are written to `<workspace>/metrics.txt`; the workspace directory is created automatically.

| Checkpoint | Eval preset |
|------------|-------------|
| `dl3dv_2v.safetensors` | `eval_dl3dv_2view` |
| `dl3dv_4v.safetensors` | `eval_dl3dv_4view` |
| `dl3dv_6v.safetensors` | `eval_dl3dv_6view` |

**Example (6-view preset):**

```bash
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_6view \
    --workspace results/dl3dv_eval/6view \
    --resume checkpoints/dl3dv_6v.safetensors \
    --use_ttt_for_eval \
    --eval_n_media_dumps 20 \
```

Presets `eval_dl3dv_2view` and `eval_dl3dv_4view` select the matching evaluation JSONs. Remove `--use_ttt_for_eval` to turn off test-time token tuning.

**Media dumps:** `--eval_n_media_dumps N` writes PNGs, MP4s, depth vis, and PLY for the first `N` dataloader batches under `<workspace>/{images,videos,depths,gaussians}/` (default `0` = metrics only).


## Training

**1. Base run** (`train_dl3dv_base` preset):

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m tokengs.train train_dl3dv_base \
    --workspace workspace/dl3dv_base \
    --experiment_name dl3dv_base
```

**2. Finetune** from a checkpoint (presets `finetune_dl3dv_2view`, `finetune_dl3dv_4view`, `finetune_dl3dv_6view`):

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m tokengs.train finetune_dl3dv_2view \
    --workspace workspace/dl3dv_2view \
    --experiment_name dl3dv_2view \
    --resume workspace/dl3dv_base/model.safetensors
```

Swap the subcommand for 4- or 6-view finetune presets as needed.

## Kubric Dynamic Model

A Kubric dynamic checkpoint is available at `checkpoints_26.06/kubric_dyn.safetensors`. Point `data/kubric` at the Kubric4D dataset:

<img src="assets/kubric_dynamic_static_scene_002.gif" alt="Kubric dynamic model visualization: all, dynamic-only, and static-only renders for scene 002" width="512">

```text
data/kubric/
  v0/
    <scene>/
      output_000.tar
      output_001.tar
      ...
  v1/
  v2/
  ...
```

The top-level `v0`, `v1`, `v2`, ... folders are data splits. Each `output_{view:03d}.tar` contains `metadata.json`, `rgba_{frame:05d}.png`, and `depth_{frame:05d}.tiff`. The dynamic presets use pointmap camera scaling, so input-frame depth TIFFs are loaded for scale normalization.

### Warm-Start Strategy

The released dynamic preset warm-starts static GS tokens from the DL3DV base checkpoint and initializes dynamic GS tokens from the static tokens. Its auxiliary dynamic-only render loss is held at `0.3` for 5K steps, then linearly decays to `0` over the next 10K steps.

Finetune example:

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m tokengs.train finetune_dl3dv_kubric_dyn_release \
    --workspace workspace/kubric_dyn \
    --experiment_name kubric_dyn \
    --resume checkpoints/dl3dv_base.safetensors
```

The render script uses the released Kubric preset by default; pass `--workspace` only when rendering a finetuned workspace with its own `config.yaml`.

```bash
python scripts/render_kubric_dyn.py \
    --preset finetune_dl3dv_kubric_dyn_release \
    --ckpt checkpoints_26.06/kubric_dyn.safetensors \
    --kubric_root data/kubric \
    --out_dir results/kubric_dyn_renders \
    --scene_idx 0 \
    --n_scenes 4 \
    --fps 8
```

For each scene, the render script writes fixed-camera, trajectory, dynamic-only, static-only, and input-camera hstack videos.

## License

TokenGS is released under the [Apache License 2.0](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Citation

If you use TokenGS in your research, please cite:

```bibtex
@article{tokengs2026,
  title={TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens},
  author={Jiawei Ren and Michal Tyszkiewicz and Jiahui Huang and Zan Gojcic},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
