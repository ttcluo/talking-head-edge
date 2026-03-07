"""
MuseTalk UNet 知识蒸馏训练脚本。

Teacher: musetalkV15/unet.pth  (~850MB FP32, block_out_channels=[320,640,1280,1280])
Student: student_musetalk.json (~136MB FP32, block_out_channels=[128,256,512,512])

蒸馏损失：
  L_total = λ1 * L_output   (MSE student_out vs teacher_out, 最核心)
          + λ2 * L_feat     (cosine_sim 中间特征对齐，带 projector)
          + λ3 * L_recon    (L1 student_out vs gt_latent)
          + λ4 * L_vgg      (VGG 感知损失)
          + λ5 * L_sync     (SyncNet 唇同步损失)

运行方式（在 $MUSE_ROOT 目录下）：
  accelerate launch --num_processes 4 $REPO/step3/distill/train_distill.py \
      --config $REPO/step3/distill/configs/distill.yaml \
      --student_config $REPO/step3/distill/configs/student_musetalk.json
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import WhisperModel

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.utils.utils import seed_everything, process_audio_features
from musetalk.utils.training_utils import (
    initialize_dataloaders,
    initialize_syncnet,
    initialize_vgg,
)

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")


# ==================== 特征提取 Hook ====================

class FeatureHook:
    def __init__(self):
        self._feats: dict = {}
        self._handles = []

    def register(self, module: nn.Module, name: str):
        def hook(_, __, output):
            # down_block 输出是 (hidden_states, res_samples) tuple，取第一个
            self._feats[name] = output[0] if isinstance(output, tuple) else output
        self._handles.append(module.register_forward_hook(hook))

    def get(self, name: str):
        return self._feats.get(name)

    def clear(self):
        self._feats.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# ==================== 特征投影器 ====================

class FeatProjector(nn.Module):
    """将 student 特征投影到 teacher 特征空间（1×1 Conv）。"""

    def __init__(self, student_channels: list, teacher_channels: list):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(s, t, 1, bias=False)
            for s, t in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_feats: list) -> list:
        return [proj(f) for proj, f in zip(self.projs, student_feats)]


# ==================== 主函数 ====================

def main(args):
    cfg = OmegaConf.load(args.config)

    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs()
    pg_kwargs  = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=["tensorboard"],
        project_dir=os.path.join(save_dir, "tensorboard"),
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    seed_everything(cfg.seed)

    weight_dtype = torch.float32
    if cfg.solver.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.solver.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ==================== Teacher（冻结）====================
    logger.info("加载 Teacher UNet（冻结）...")
    with open(cfg.teacher_unet_config) as f:
        teacher_cfg = json.load(f)
    teacher_unet = UNet2DConditionModel(**teacher_cfg)
    ckpt = torch.load(cfg.teacher_unet_path, map_location="cpu")
    teacher_unet.load_state_dict(ckpt, strict=False)
    teacher_unet.set_attn_processor(AttnProcessor())
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()
    logger.info(f"Teacher 参数量: {sum(p.numel() for p in teacher_unet.parameters())/1e6:.1f}M")

    # ==================== Student（训练目标）====================
    logger.info("初始化 Student UNet...")
    student_config_path = args.student_config or os.path.join(
        os.path.dirname(__file__), "configs", "student_musetalk.json"
    )
    with open(student_config_path) as f:
        student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    student_unet = UNet2DConditionModel(**student_cfg)
    student_unet.set_attn_processor(AttnProcessor())
    if cfg.solver.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
    logger.info(f"Student 参数量: {sum(p.numel() for p in student_unet.parameters())/1e6:.1f}M")

    # ==================== VAE & Whisper（冻结）====================
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder=cfg.vae_type
    )
    vae.requires_grad_(False)

    wav2vec = WhisperModel.from_pretrained(cfg.whisper_path).eval()
    wav2vec.requires_grad_(False)

    # ==================== 特征 Hook ====================
    DISTILL_LAYERS = list(cfg.feat_distill_layers)
    teacher_hook = FeatureHook()
    student_hook = FeatureHook()

    def _get_module(unet, path: str):
        m = unet
        for p in path.split("."):
            m = getattr(m, p)
        return m

    for layer_name in DISTILL_LAYERS:
        teacher_hook.register(_get_module(teacher_unet, layer_name), layer_name)
        student_hook.register(_get_module(student_unet, layer_name), layer_name)

    # Teacher/Student 各层通道数（与 block_out_channels 对应）
    teacher_ch = [320, 640, 1280, 640, 320]
    student_ch  = [128, 256,  512, 256, 128]
    projector = FeatProjector(student_ch, teacher_ch)

    # ==================== 辅助损失 ====================
    lp = cfg.loss_params
    syncnet = initialize_syncnet(cfg, accelerator) if lp.sync_loss > 0 else None
    vgg_fn  = initialize_vgg(cfg, accelerator)    if lp.vgg_loss  > 0 else None

    # ==================== 数据 ====================
    dataloader_dict = initialize_dataloaders(cfg)
    train_dataloader = dataloader_dict["train_dataloader"]

    # ==================== 优化器 ====================
    trainable_params = list(student_unet.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps,
    )

    # ==================== Accelerate 准备 ====================
    (
        student_unet, projector, teacher_unet,
        vae, wav2vec, optimizer,
        train_dataloader, lr_scheduler,
    ) = accelerator.prepare(
        student_unet, projector, teacher_unet,
        vae, wav2vec, optimizer,
        train_dataloader, lr_scheduler,
    )
    teacher_unet.to(weight_dtype)
    vae.to(weight_dtype)
    wav2vec.to(weight_dtype)
    student_unet.to(weight_dtype)

    global_step = 0
    progress_bar = tqdm(
        range(cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    logger.info("=" * 60)
    logger.info("  开始蒸馏训练")
    logger.info("=" * 60)

    for _ in range(10000):
        student_unet.train()
        projector.train()

        for batch in train_dataloader:
            with accelerator.accumulate(student_unet):

                # ---------- 数据准备（与 train.py 完全一致）----------
                pixel_values = batch["pixel_values_vid"].to(
                    weight_dtype, non_blocking=True
                )
                ref_values = batch["pixel_values_ref_img"].to(
                    weight_dtype, non_blocking=True
                )
                bsz, num_frames, c, h, w = pixel_values.shape

                # 音频特征（经 Whisper encoder 处理）
                audio_prompts = process_audio_features(
                    cfg, batch, wav2vec, bsz, num_frames, weight_dtype
                )
                # reshape 与 train.py 相同
                audio_flat = rearrange(audio_prompts, "b f c h w -> (b f) c h w")
                audio_flat = rearrange(audio_flat, "(b f) c h w -> (b f) (c h) w", b=bsz)

                # VAE encode（无梯度）
                with torch.no_grad():
                    frames = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    gt_lat = vae.encode(frames).latent_dist.mode()
                    gt_lat = gt_lat * vae.config.scaling_factor

                    masked = pixel_values.clone()
                    masked[:, :, :, h // 2:, :] = -1
                    masked_frames = rearrange(masked, "b f c h w -> (b f) c h w")
                    masked_lat = vae.encode(masked_frames).latent_dist.mode()
                    masked_lat = masked_lat * vae.config.scaling_factor

                    ref_frames = rearrange(ref_values, "b f c h w -> (b f) c h w")
                    ref_lat = vae.encode(ref_frames).latent_dist.mode()
                    ref_lat = ref_lat * vae.config.scaling_factor

                input_lat = torch.cat([masked_lat, ref_lat], dim=1).to(weight_dtype)
                timestep  = torch.zeros(
                    input_lat.shape[0], dtype=torch.long, device=input_lat.device
                )

                # ---------- Teacher 推理 ----------
                teacher_hook.clear()
                with torch.no_grad():
                    teacher_out = teacher_unet(
                        input_lat, timestep,
                        encoder_hidden_states=audio_flat,
                        return_dict=False,
                    )[0]
                teacher_feats = [teacher_hook.get(l) for l in DISTILL_LAYERS]

                # ---------- Student 推理 ----------
                student_hook.clear()
                student_out = student_unet(
                    input_lat, timestep,
                    encoder_hidden_states=audio_flat,
                    return_dict=False,
                )[0]
                student_feats = [student_hook.get(l) for l in DISTILL_LAYERS]

                # ---------- 损失 ----------
                # 1. 输出蒸馏
                L_out = F.mse_loss(student_out, teacher_out.detach())

                # 2. 特征蒸馏
                proj_feats = projector(student_feats)
                L_feat = torch.tensor(0.0, device=input_lat.device)
                valid_layers = 0
                for pf, tf in zip(proj_feats, teacher_feats):
                    if tf is not None and pf is not None:
                        sim = F.cosine_similarity(
                            pf.flatten(2), tf.detach().flatten(2), dim=1
                        )
                        L_feat = L_feat + (1 - sim.mean())
                        valid_layers += 1
                if valid_layers > 0:
                    L_feat = L_feat / valid_layers

                # 3. GT L1 重建
                L_recon = F.l1_loss(student_out, gt_lat.detach())

                # 4. VGG 感知
                L_vgg = torch.tensor(0.0, device=input_lat.device)
                if vgg_fn is not None and lp.vgg_loss > 0:
                    with torch.no_grad():
                        pred_img = vae.decode(
                            student_out / vae.config.scaling_factor
                        ).sample.float()
                        gt_img = vae.decode(
                            gt_lat / vae.config.scaling_factor
                        ).sample.float()
                    L_vgg = vgg_fn(pred_img, gt_img)

                # 5. SyncNet
                L_sync = torch.tensor(0.0, device=input_lat.device)
                if syncnet is not None and lp.sync_loss > 0:
                    try:
                        height = pixel_values.shape[3]
                        pred_img_sync = vae.decode(
                            student_out / vae.config.scaling_factor
                        ).sample
                        pred_img_sync = pred_img_sync[:, :, height // 2:, :]
                        audio_embed = syncnet.get_audio_embed(batch["mel"])
                        vision_embed = syncnet.get_vision_embed(
                            pred_img_sync.reshape(bsz, -1, *pred_img_sync.shape[2:])
                        )
                        L_sync = 1 - F.cosine_similarity(
                            audio_embed, vision_embed, dim=1
                        ).mean()
                    except Exception:
                        pass

                loss = (
                    lp.output_distill * L_out
                    + lp.feat_distill  * L_feat
                    + lp.l1_recon      * L_recon
                    + lp.vgg_loss      * L_vgg
                    + lp.sync_loss     * L_sync
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, cfg.solver.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---------- 日志 & 保存 ----------
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss":    f"{loss.item():.4f}",
                    "L_out":   f"{L_out.item():.4f}",
                    "L_feat":  f"{L_feat.item():.4f}",
                    "L_recon": f"{L_recon.item():.4f}",
                })
                if accelerator.is_main_process:
                    accelerator.log({
                        "loss":            loss.item(),
                        "L_output_distill": L_out.item(),
                        "L_feat_distill":  L_feat.item(),
                        "L_recon":         L_recon.item(),
                        "L_vgg":           L_vgg.item(),
                        "L_sync":          L_sync.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    raw = accelerator.unwrap_model(student_unet)
                    ckpt_path = os.path.join(save_dir, f"student_unet-{global_step}.pth")
                    torch.save(raw.state_dict(), ckpt_path)
                    logger.info(f"✓ checkpoint: {ckpt_path}")

                if global_step >= cfg.solver.max_train_steps:
                    break

        if global_step >= cfg.solver.max_train_steps:
            break

    # ==================== 最终保存 ====================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(student_unet)
        final_path = os.path.join(save_dir, "student_unet_final.pth")
        torch.save(raw.state_dict(), final_path)
        n = sum(p.numel() for p in raw.parameters())
        logger.info(f"✓ 训练完成: {final_path}")
        logger.info(f"  参数量: {n/1e6:.1f}M  FP32: ~{n*4/1e6:.0f}MB  INT8: ~{n/1e6:.0f}MB")

    teacher_hook.remove()
    student_hook.remove()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str, required=True,
                        help="蒸馏训练配置（distill.yaml）")
    parser.add_argument("--student_config",
                        type=str, default=None,
                        help="Student UNet 架构 JSON（默认 configs/student_musetalk.json）")
    args = parser.parse_args()
    main(args)
