# TokenGS

![Teaser: TokenGS results and exploration](assets/tokengs_explore.gif)

**TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens** <br>
Jiawei Ren*, Michal Tyszkiewicz*, Jiahui Huang†, Zan Gojcic† <br>
\* indicates equal contribution, † indicates equal advising

[**Paper**](https://arxiv.org/abs/2604.15239) · [**Project Page**](https://research.nvidia.com/labs/toronto-ai/tokengs/) · [**HuggingFace**](https://huggingface.co/jiaweir/tokengs)

TokenGS predicts 3D Gaussians with a self-supervised rendering objective. An encoder–decoder stacks learnable Gaussian tokens so the number of primitives is not tied to image resolution or view count.

## Installation

Install the package in editable mode (dependencies include PyTorch, gsplat, and [fused-ssim](https://github.com/rahul-goel/fused-ssim) via `pyproject.toml`):

```bash
uv pip install -e .
```

**Environment:** Python 3.11, CUDA 12.6+ (see `pyproject.toml` for pinned versions).

**Data:** DL3DV layout, symlinks, and `dataset_kwargs` are described in **[data/DATA.md](data/DATA.md)**.

## Evaluation

Place weights under `checkpoints/` (or pass any path to `--resume`). Metrics are written to `<workspace>/metrics.txt`; the workspace directory is created automatically.

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
