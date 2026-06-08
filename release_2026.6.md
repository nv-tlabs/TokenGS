# Release 2026.6

![TokenGS 2026.6 sweep comparison](assets/sweep_r26.06.gif)

This note documents the June 2026 TokenGS model updates.

Model weights are available on Hugging Face under [checkpoints_26.06](https://huggingface.co/jiaweir/tokengs/tree/main/checkpoints_26.06).

## Overview

- **Latent bottleneck models:** new 2-, 4-, and 6-view DL3DV checkpoints with a 12-layer encoder, multiscale encoder readout, and 4096 learned scene latent tokens.
- **Scene-latent tuning:** latent bottleneck eval presets use `scene-latent-tuning`, which optimizes per-scene latent tokens at test time instead of Gaussian tokens.
- **Mean-of-gradients training:** `mean_of_grads` adds per-scene and per-view chunked rendering modes for memory-constrained finetuning while matching the standard training objective.

## Latent Bottleneck Models

The latent bottleneck models keep the TokenGS Gaussian-token decoder but insert a learned latent bottleneck between the image encoder and the Gaussian decoder. The release architecture uses:

- 12-layer encoder.
- Multiscale encoder readout from layers 5, 7, 9, and 11.
- 4096 learned latent bottleneck tokens.

The released DL3DV variants cover 2, 4, and 6 input views. For each view count, two loss variants are provided:

- SSIM variant: `0.8 * L1 + 0.2 * SSIM`.
- LPIPS variant: `L2 + 0.5 * LPIPS`.

Evaluation presets use strict checkpoint loading by default, so a checkpoint fails fast if paired with the wrong architecture or view-count preset.
For the latent bottleneck eval presets, test-time tuning uses `scene-latent-tuning`: the model first encodes the input views into scene latent bottleneck tokens, then optimizes those per-scene latents against the input views before rendering the benchmark target views.

## Mean-Of-Gradients Training

This release includes the mean-of-gradients training mode from the main branch. It renders smaller scene/view chunks, accumulates the mean Gaussian gradients, and backpropagates the averaged gradient through the full model. This is useful for memory-constrained finetuning with the same objective as standard training.

Mechanically, the model first predicts the full Gaussian set once, then detaches those Gaussians and renders the target scenes/views in smaller chunks. Each chunk computes the same render loss and its gradient with respect to the Gaussian parameters; those chunk gradients are accumulated and used as a surrogate `backward_loss` so autograd propagates the averaged Gaussian gradient back through the original network. This keeps the expensive renderer activations scoped to one chunk at a time while preserving the full-batch training signal.

Set `mean_of_grads` to `per-scene` or `per-view` to enable it. `mean_of_grads_scene_chunk_size` controls the number of scenes per render chunk, and `mean_of_grads_view_chunk_size` can override the number of target views per chunk. The mode is mutually exclusive with `deferred_bp`.

## Checkpoints And Presets

| Hugging Face path | Local checkpoint name | Eval preset |
|-------------------|-----------------------|-------------|
| `checkpoints_26.06/dl3dv_latent_2v_ssim.safetensors` | `dl3dv_latent_2v_ssim.safetensors` | `eval_dl3dv_latent_2view_ssim` |
| `checkpoints_26.06/dl3dv_latent_4v_ssim.safetensors` | `dl3dv_latent_4v_ssim.safetensors` | `eval_dl3dv_latent_4view_ssim` |
| `checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors` | `dl3dv_latent_6v_ssim.safetensors` | `eval_dl3dv_latent_6view_ssim` |
| `checkpoints_26.06/dl3dv_latent_2v_lpips.safetensors` | `dl3dv_latent_2v_lpips.safetensors` | `eval_dl3dv_latent_2view_lpips` |
| `checkpoints_26.06/dl3dv_latent_4v_lpips.safetensors` | `dl3dv_latent_4v_lpips.safetensors` | `eval_dl3dv_latent_4view_lpips` |
| `checkpoints_26.06/dl3dv_latent_6v_lpips.safetensors` | `dl3dv_latent_6v_lpips.safetensors` | `eval_dl3dv_latent_6view_lpips` |

Example:

```bash
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_latent_6view_lpips \
    --workspace results/dl3dv_latent_eval/6view_lpips \
    --resume checkpoints/dl3dv_latent_6v_lpips.safetensors \
    --use_ttt_for_eval \
    --eval_n_media_dumps 20
```

## Benchmark Results

DL3DV-10K benchmark, 140 scenes, fixed scene scale `0.15`. TTT numbers use 50-step scene latent tuning with LR `1e-2`.

| Views | Checkpoint variant | Eval preset | w/o TTT PSNR | w/o TTT SSIM | w/o TTT LPIPS | w/ TTT PSNR | w/ TTT SSIM | w/ TTT LPIPS |
|-------|--------------------|-------------|--------------|--------------|---------------|-------------|-------------|--------------|
| 2 | SSIM | `eval_dl3dv_latent_2view_ssim` | 20.493 | 0.658 | 0.381 | 20.899 | 0.668 | 0.333 |
| 4 | SSIM | `eval_dl3dv_latent_4view_ssim` | 24.012 | 0.778 | 0.277 | 25.408 | 0.822 | 0.213 |
| 6 | SSIM | `eval_dl3dv_latent_6view_ssim` | 25.120 | 0.806 | 0.257 | 27.304 | 0.866 | 0.185 |
| 2 | LPIPS | `eval_dl3dv_latent_2view_lpips` | 19.772 | 0.620 | 0.317 | 20.093 | 0.634 | 0.287 |
| 4 | LPIPS | `eval_dl3dv_latent_4view_lpips` | 23.089 | 0.754 | 0.215 | 24.089 | 0.789 | 0.157 |
| 6 | LPIPS | `eval_dl3dv_latent_6view_lpips` | 24.077 | 0.783 | 0.194 | 25.505 | 0.829 | 0.121 |

Higher is better for PSNR and SSIM. Lower is better for LPIPS.
