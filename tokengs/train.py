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

import tyro
import time
import os
import json
import datetime
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from safetensors.torch import load_file

import imageio
import numpy as np

from tokengs.options import AllConfigs
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry

import warnings

from tokengs.utils.gaussians import Gaussians
warnings.filterwarnings("ignore")


def setup_workspace_and_status(opt, accelerator):
    """Setup workspace directory and check for existing completion."""
    status_dir = os.path.join(opt.workspace, "status")
    complete_file = os.path.join(status_dir, "COMPLETE")
    
    if accelerator.is_main_process:
        os.makedirs(status_dir, exist_ok=True)
        if os.path.exists(complete_file):
            raise RuntimeError(f"Found existing COMPLETE file at {complete_file}, remove it if you want to run the job again")


def load_checkpoint_and_resume(opt, accelerator):
    """Load checkpoint and resume training state."""
    epoch_start = 0
    wandb_run_id = None
    
    if os.path.exists(f'{opt.workspace}/model.safetensors') and os.path.exists(f'{opt.workspace}/metadata.json'):
        if accelerator.is_main_process:
            print(f"Resuming from {opt.workspace}/model.safetensors")
        opt.resume = f'{opt.workspace}/model.safetensors'
        
        with open(f'{opt.workspace}/metadata.json', 'r') as f:
            dc = json.load(f)
            epoch_start = dc['epoch'] + 1
            if 'wandb_run_id' in dc:
                wandb_run_id = dc['wandb_run_id']
    
    return epoch_start, wandb_run_id


def setup_wandb(opt, accelerator, epoch_start, wandb_run_id):
    """Setup wandb logging."""
    if not accelerator.is_main_process or not opt.use_wandb:
        return None, None
    
    import wandb
    run_name = datetime.datetime.now().strftime("%b %d, %I:%M%p")
    
    # Initialize wandb - resume if we have a run_id, otherwise create new run
    if wandb_run_id and epoch_start > 0:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  id=wandb_run_id, resume="must")
        print(f"Resuming wandb run {wandb_run_id}")
    else:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  name=f"{opt.experiment_name} {run_name}")
        
    # Store wandb run id for potential future resuming
    wandb_run_id = wandb.run.id
    print(f"wandb run id: {wandb.run.id}")
    
    tensorboard_root_dir = f'{opt.out_dir}/{opt.experiment_name}' if opt.experiment_name else None
    wandb.tensorboard.patch(root_logdir=tensorboard_root_dir, save=False)
    writer = SummaryWriter(log_dir=tensorboard_root_dir)
    print(f"tensorboard root dir: {tensorboard_root_dir}")
    
    return wandb_run_id, writer


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    """Load model checkpoint with tolerance for shape mismatches."""
    if opt.resume is None or opt.resume == 'None':
        return
    
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    
    # tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    if opt.init_tokens_from_existing and epoch_start == 0:
        _initialize_tokens_from_existing(ckpt, state_dict, accelerator)
    
    if opt.init_dynamic_tokens_from_static and epoch_start == 0 and 'gs_tokens' in ckpt:
        _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator)


def _initialize_tokens_from_existing(ckpt, state_dict, accelerator):
    """Initialize tokens from existing checkpoint."""
    with torch.no_grad():
        for token_type in ['gs_tokens', 'gs_tokens_dynamic']:
            if token_type not in ckpt:
                continue
            pretrained_tokens = ckpt[token_type]
            current_tokens = state_dict[token_type]    
            N_old = pretrained_tokens.shape[0]
            N_new = current_tokens.shape[0]
            
            if N_new != N_old:
                accelerator.print(f'[INFO] Initializing {token_type} from pretrained tokens, N_old: {N_old}, N_new: {N_new}')
            
            if N_new <= N_old:
                # Downsample / take subset if fewer tokens
                idx = torch.linspace(0, N_old - 1, N_new).long()
                current_tokens.copy_(pretrained_tokens[idx])
            else:
                # Copy existing tokens first
                current_tokens[:N_old].copy_(pretrained_tokens)

                # Initialize additional tokens by sampling from pretrained tokens + noise
                extra_tokens = current_tokens[N_old:]
                repeat_factor = (extra_tokens.shape[0] + N_old - 1) // N_old

                expanded = pretrained_tokens.repeat((repeat_factor, 1))[:extra_tokens.shape[0]]
                noise = 0.01 * torch.randn_like(expanded)   # small perturbation
                extra_tokens.copy_(expanded + noise)


def _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator):
    """Initialize dynamic tokens from static tokens."""
    accelerator.print(f'[INFO] Initializing dynamic tokens from static tokens')
    
    with torch.no_grad():
        static_tokens = ckpt["gs_tokens"]  # shape: [N_static, D]
        dynamic_tokens = state_dict['gs_tokens_dynamic']          # shape: [N_dynamic, D]
        N_dynamic = dynamic_tokens.shape[0]
        N_static = static_tokens.shape[0]

        if N_dynamic == N_static:
            # direct copy
            dynamic_tokens.copy_(static_tokens)
        elif N_dynamic > N_static:
            # replicate or pad
            repeat_factor = (N_dynamic + N_static - 1) // N_static
            dynamic_tokens.copy_(
                static_tokens.repeat((repeat_factor, 1))[:N_dynamic]
            )
        else:
            # random subset if fewer dynamic tokens
            idx = torch.randperm(N_static)[:N_dynamic]
            dynamic_tokens.copy_(static_tokens[idx])


def setup_optimizer(opt, model, accelerator, epoch_start):
    """Setup optimizer. Call before accelerator.prepare()."""
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = []
    if len(decay_params) > 0:
        optim_groups.append({'params': decay_params, 'weight_decay': 0.05})
    if len(nodecay_params) > 0:
        optim_groups.append({'params': nodecay_params, 'weight_decay': 0.0})

    optimizer = torch.optim.AdamW(optim_groups, lr=opt.lr, betas=(0.9, 0.95), fused=True)

    if epoch_start > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, 'optimizer.pth'), map_location='cpu'))

    return optimizer


def setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start):
    """Setup scheduler. Call after accelerator.prepare() with per-GPU iters_per_epoch."""
    steps_per_epoch = iters_per_epoch // opt.gradient_accumulation_steps
    total_steps = opt.num_epochs * steps_per_epoch
    pct_start = min(opt.pct_start_steps / total_steps, 0.99)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt.lr, total_steps=total_steps,
        pct_start=pct_start, final_div_factor=opt.final_div_factor,
    )

    if epoch_start > 0:
        scheduler.load_state_dict(torch.load(os.path.join(opt.workspace, 'scheduler.pth')))

    return scheduler


def save_checkpoint(opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id):
    """Save model checkpoint and metadata."""
    accelerator.wait_for_everyone()
    accelerator.save_model(model, opt.workspace)
    
    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), os.path.join(opt.workspace, 'optimizer.pth'))
        torch.save(scheduler.state_dict(), os.path.join(opt.workspace, 'scheduler.pth'))
        
        metadata = {'epoch': epoch}
        if wandb_run_id:
            metadata['wandb_run_id'] = wandb_run_id
            
        with open(f'{opt.workspace}/metadata.json', 'w') as f:
            json.dump(metadata, f)


def log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=True):
    """Log training/evaluation images or videos."""
    if not accelerator.is_main_process:
        return
        
    prefix = "train" if is_train else "eval"
    
    # Ensure images directory exists
    os.makedirs(f'{opt.workspace}/images', exist_ok=True)
    
    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = np.clip(out['images_pred'].detach().cpu().numpy(), 0, 1) # [B, V, 3, output_size, output_size]
    
    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[4], 3) # [B*output_size, V*output_size, 3]
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_gt_images_{epoch}_{i}.jpg', (np.clip(gt_images, 0, 1) * 255).astype(np.uint8))

    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_pred_images_{epoch}_{i}.jpg', (np.clip(pred_images, 0, 1) * 255).astype(np.uint8))

    if opt.use_wandb:
        writer.add_image(f'image/{prefix}_gt', gt_images.clip(0,1.0), epoch, dataformats='HWC')
        writer.add_image(f'image/{prefix}_pred', pred_images.clip(0,1.0), epoch, dataformats='HWC')


def train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                iters_per_epoch, epoch, writer, start_time, train_dataset):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    log_time = time.time()

    def train_step(data, should_log_images, iteration):
        """Execute a single training step and return metrics."""
        optimizer.zero_grad()

        out = model(data)
        loss = out['loss']
        psnr = out['psnr']
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

        optimizer.step()
        scheduler.step()

        loss_value = loss.detach()
        psnr_value = psnr.detach()
        loss_value_detailed = {
            'loss_mse': out['loss_mse'].detach(),
        }
        if 'loss_ssim' in out:
            loss_value_detailed['loss_ssim'] = out['loss_ssim'].detach()
        if 'loss_visibility' in out:
            loss_value_detailed['loss_visibility'] = out['loss_visibility'].detach()
        
        if should_log_images:
            log_training_images(opt, accelerator, data, out, epoch, iteration, writer, is_train=True)
        
        return loss_value, loss_value_detailed, psnr_value

    train_dataset.set_rng_epoch(epoch)
    if accelerator.is_main_process:
        print(f"[INFO] Setting RNG epoch to {epoch}")

    for i, data in enumerate(iter(train_dataloader)):
        if i >= opt.max_iters_per_epoch:
            break

        # Determine if we need to log images this iteration
        should_log_images = accelerator.is_main_process and (i % opt.log_image_freq == 0)
            
        with accelerator.accumulate(model):
            loss_value, loss_value_detailed, psnr_value = train_step(data, should_log_images, i)
            total_loss += loss_value
            total_psnr += psnr_value

            if opt.use_wandb and accelerator.is_main_process:
                global_step = epoch * iters_per_epoch + i
                writer.add_scalar(f"psnr/train_iteration", psnr_value.item(), global_step)
                writer.add_scalar(f"loss/train_iteration", loss_value.item(), global_step)
                writer.add_scalar(f"lr/train_iteration", scheduler.get_last_lr()[0], global_step)
                try:
                    writer.add_scalar(f"time/train_iteration", time.time() - start_time, global_step)
                    for key, value in loss_value_detailed.items():
                        writer.add_scalar(f"loss_detailed/{key}/train_iteration", value.item(), global_step)
                except: 
                    pass

        if accelerator.is_main_process:
            # logging
            if i % opt.print_freq == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()    
                elapsed = time.time() - log_time
                speed = opt.print_freq / elapsed if elapsed > 0 else 0
                print(f"[INFO] {i}/{iters_per_epoch} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.10f} loss: {loss_value.item():.6f} speed: {speed:.2f} it/s")
                log_time = time.time()

    total_loss = accelerator.gather_for_metrics(total_loss).mean()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
    
    if accelerator.is_main_process:
        total_loss /= iters_per_epoch
        total_psnr /= iters_per_epoch
        accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

        if opt.use_wandb:
            writer.add_scalar(f"psnr/train", total_psnr.item(), epoch)
            writer.add_scalar(f"loss/train", total_loss.item(), epoch)


def log_gaussian_histograms(opt, all_gaussians, epoch, writer):
    """Log histograms of Gaussian properties to wandb."""
    # Concatenate all gaussians: [B, N, 14] -> [B*N, 14]
    gaussians = Gaussians.from_raw(torch.cat(all_gaussians, dim=0).reshape(-1, 14))
    
    # Log position histograms (x, y, z separately)
    # Clamp to bin range so out-of-bounds values appear in extreme bins
    pos_bins = torch.linspace(-25, 25, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/pos_{label}', gaussians.xyz[:, i].clamp(-25, 25), global_step=epoch, bins=pos_bins)
    
    # Log opacity histogram [0, 1]
    opacity_bins = torch.linspace(0, 1, 100)
    writer.add_histogram('gaussian/opacity', gaussians.opacity.flatten().clamp(0, 1), global_step=epoch, bins=opacity_bins)
    
    scale_bins = torch.linspace(0, opt.gaussian_scale_cap, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/scale_{label}', gaussians.scaling[:, i].clamp(0, opt.gaussian_scale_cap), global_step=epoch, bins=scale_bins)
    
    # Log rotation histograms [-1, 1] (quaternion components)
    rotation_bins = torch.linspace(-1, 1, 100)
    for i, label in enumerate("wxyz"):
        writer.add_histogram(f'gaussian/rotation_{label}', gaussians.rotation[:, i].clamp(-1, 1), global_step=epoch, bins=rotation_bins)
    
    # Log RGB histograms [0, 1]
    rgb_bins = torch.linspace(0, 1, 100)
    for i, label in enumerate("rgb"):
        writer.add_histogram(f'gaussian/rgb_{label}', gaussians.rgb[:, i].clamp(0, 1), global_step=epoch, bins=rgb_bins)


def evaluate_epoch(opt, accelerator, model, test_dataloader, epoch, writer):
    """Evaluate for one epoch."""
    use_input_supervision = opt.use_input_supervision
    opt.use_input_supervision = False
    with torch.inference_mode():
        model.eval()

        total_psnr = 0
        all_gaussians = []
        for i, data in enumerate(iter(test_dataloader)):
            out = model(data)
            psnr = out['psnr']
            total_psnr += psnr.detach()
            
            # Collect gaussians for histogram logging
            if accelerator.is_main_process:
                all_gaussians.append(out['gaussians'].detach().cpu())
            
            # save some images
            log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=False)

        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")

            if opt.use_wandb:
                writer.add_scalar(f"psnr/eval", total_psnr.item(), epoch)
                
                # Log Gaussian property histograms
                if len(all_gaussians) > 0:
                    log_gaussian_histograms(opt, all_gaussians, epoch, writer)

    opt.use_input_supervision = use_input_supervision


def main():    
    start_time = time.time()
    opt = tyro.cli(AllConfigs)

    torch.manual_seed(opt.seed)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    # Setup workspace and status
    setup_workspace_and_status(opt, accelerator)

    # Load checkpoint and resume
    epoch_start, wandb_run_id = load_checkpoint_and_resume(opt, accelerator)

    # Setup wandb
    wandb_run_id, writer = setup_wandb(opt, accelerator, epoch_start, wandb_run_id)
    
    if accelerator.is_main_process:
        print(opt)

        config_save_path = os.path.join(opt.workspace, "config.yaml")
        if not os.path.exists(config_save_path):
            with open(config_save_path, "w") as f:
                f.write(tyro.extras.to_yaml(opt))
            print(f"[INFO] Config saved to {config_save_path=}")

    # model
    model = model_registry[opt.model_type](opt)

    # Load model checkpoint
    load_model_checkpoint(opt, model, accelerator, epoch_start)
    
    # Data
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_multi_dataloader(opt, accelerator)

    # Optimizer (before prepare)
    optimizer = setup_optimizer(opt, model, accelerator, epoch_start)

    # accelerate (shards dataloader across GPUs)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Compute per-GPU iterations from the prepared (sharded) dataloader
    iters_per_epoch = min(len(train_dataloader), opt.max_iters_per_epoch)

    # Scheduler (after prepare, so iters_per_epoch is correct)
    # NOTE: do NOT call accelerator.prepare(scheduler) here -- total_steps is already
    # computed from the per-GPU iters_per_epoch, and prepare() would divide it again
    # by num_processes, causing the LR to decay to zero far too early.
    scheduler = setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start)

    # loop
    os.makedirs(opt.workspace, exist_ok=True)

    evaluate_epoch(opt, accelerator, model, test_dataloader, epoch_start-1, writer)
    
    epoch = epoch_start
    while epoch < opt.num_epochs:
        # train
        train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                    iters_per_epoch, epoch, writer, start_time, train_dataset)
        
        # checkpoint
        save_checkpoint(opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id)

        # eval
        evaluate_epoch(opt, accelerator, model, test_dataloader, epoch, writer)

        epoch += 1
            
    # If we get here, we've completed all epochs
    # Signal successful completion
    if accelerator.is_main_process:
        status_dir = os.path.join(opt.workspace, "status")
        with open(os.path.join(status_dir, "COMPLETE"), "w") as f:
            f.write(f"Training completed successfully after {opt.num_epochs} epochs")
        print(f"[INFO] Training completed successfully after {opt.num_epochs} epochs")
    
    accelerator.wait_for_everyone()
    return


if __name__ == "__main__":
    main()