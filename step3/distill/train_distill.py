"""
MuseTalk UNet 知识蒸馏训练脚本。

Teacher: musetalkV15/unet.pth  (~850MB FP32, block_out_channels=[320,640,1280,1280])
Student: student_musetalk.json (~210MB FP32, block_out_channels=[128,256,512,512])

蒸馏损失：
  L_total = λ1 * L_output   (MSE student_out vs teacher_out)
          + λ2 * L_feat     (cosine_sim 中间特征对齐，带 projector)
          + λ3 * L_recon    (L1 student_out vs gt_latent)
          + λ4 * L_vgg      (VGG 感知损失，在解码图像上)
          + λ5 * L_sync     (SyncNet 唇同步损失)

运行方式（在 $MUSE_ROOT 目录下）：
  accelerate launch --num_processes 4 $REPO/step3/distill/train_distill.py \
      --config $REPO/step3/distill/configs/distill.yaml \
      --student_config $REPO/step3/distill/configs/student_musetalk.json \
      --data_root /path/to/HDTF \
      --data_list /path/to/train.txt
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.loss.basic_loss import set_requires_grad
from musetalk.loss.syncnet import get_sync_loss
from musetalk.utils.utils import seed_everything, get_mouth_region
from musetalk.utils.training_utils import (
    initialize_dataloaders,
    initialize_syncnet,
    initialize_vgg,
)

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")


# ==================== 特征提取 Hook ====================

class FeatureHook:
    """在指定 module 的 forward 输出注册 hook，收集中间特征。"""

    def __init__(self):
        self._feats: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, module: nn.Module, name: str):
        def hook(_, __, output):
            # down_block 输出是 (hidden_states, res_samples) tuple，取 hidden_states
            self._feats[name] = output[0] if isinstance(output, tuple) else output
        self._handles.append(module.register_forward_hook(hook))

    def get(self, name: str) -> torch.Tensor | None:
        return self._feats.get(name)

    def clear(self):
        self._feats.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# ==================== 特征投影器 ====================

class FeatProjector(nn.Module):
    """将 student 特征投影到 teacher 特征空间，用于 cosine similarity 对齐。
    使用 1×1 Conv 避免改变空间分辨率。
    """

    def __init__(self, student_channels: list[int], teacher_channels: list[int]):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)
        self.projs = nn.ModuleList([
            nn.Conv2d(s, t, 1, bias=False)
            for s, t in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [proj(f) for proj, f in zip(self.projs, student_feats)]


# ==================== 主函数 ====================

def main(args):
    cfg = OmegaConf.load(args.config)

    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs()
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=["tensorboard"],
        project_dir=os.path.join(save_dir, "tensorboard"),
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    seed_everything(cfg.seed)

    weight_dtype = torch.float32
    if cfg.solver.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.solver.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ==================== 加载 Teacher（冻结）====================
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

    # ==================== 加载 Student（训练目标）====================
    logger.info("初始化 Student UNet...")
    student_config_path = args.student_config or os.path.join(
        os.path.dirname(__file__), "configs", "student_musetalk.json"
    )
    with open(student_config_path) as f:
        student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    student_unet = UNet2DConditionModel(**student_cfg)
    student_unet.set_attn_processor(AttnProcessor())
    logger.info(f"Student 参数量: {sum(p.numel() for p in student_unet.parameters())/1e6:.1f}M")

    # ==================== VAE ====================
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder=cfg.vae_type
    )
    vae.requires_grad_(False)

    # ==================== 注册 Teacher/Student 特征 Hook ====================
    DISTILL_LAYERS = cfg.feat_distill_layers
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

    # Teacher 通道数（对应 [320,640,1280,640,320] 等）/ Student 通道数
    # 由 down_blocks 和 up_blocks 的 out_channels 决定：
    #   down_blocks.0 → block_out_channels[0]
    #   down_blocks.1 → block_out_channels[1]
    #   mid_block → block_out_channels[-1]
    #   up_blocks.2 → block_out_channels[1]
    #   up_blocks.3 → block_out_channels[0]
    teacher_ch = [320, 640, 1280, 640, 320]
    student_ch = [128, 256, 512,  256, 128]
    projector = FeatProjector(student_ch, teacher_ch)

    # ==================== 损失函数 ====================
    if cfg.loss_params.sync_loss > 0:
        syncnet = initialize_syncnet(cfg, accelerator)
    else:
        syncnet = None

    if cfg.loss_params.vgg_loss > 0:
        vgg = initialize_vgg(cfg, accelerator)
    else:
        vgg = None

    # ==================== 数据 ====================
    train_dataloader = initialize_dataloaders(
        cfg,
        data_root=args.data_root,
        data_list=args.data_list,
    )

    # ==================== 优化器 ====================
    trainable_params = list(student_unet.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    num_update_steps = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.solver.max_train_steps,
        eta_min=1e-7,
    )

    # ==================== Accelerate 准备 ====================
    (
        student_unet,
        projector,
        teacher_unet,
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        student_unet, projector, teacher_unet, vae,
        optimizer, train_dataloader, lr_scheduler
    )

    teacher_unet.to(weight_dtype)
    vae.to(weight_dtype)
    student_unet.to(weight_dtype)

    global_step = 0
    progress_bar = tqdm(
        range(cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    logger.info("=" * 60)
    logger.info("  开始蒸馏训练")
    logger.info("=" * 60)

    for epoch in range(10000):
        student_unet.train()
        projector.train()

        for batch in train_dataloader:
            with accelerator.accumulate(student_unet):
                # ---------- 数据准备 ----------
                pixel_values = batch["pixel_values_vid"].to(weight_dtype)
                ref_values   = batch["pixel_values_ref_img"].to(weight_dtype)
                bsz, nf, c, h, w = pixel_values.shape

                masked = pixel_values.clone()
                masked[:, :, :, h // 2:, :] = -1

                masked_frames = rearrange(masked,      "b f c h w -> (b f) c h w")
                ref_frames    = rearrange(ref_values,  "b f c h w -> (b f) c h w")
                gt_frames     = rearrange(pixel_values,"b f c h w -> (b f) c h w")

                with torch.no_grad():
                    masked_lat = vae.encode(masked_frames).latent_dist.mode()
                    masked_lat = masked_lat * vae.config.scaling_factor
                    ref_lat    = vae.encode(ref_frames).latent_dist.mode()
                    ref_lat    = ref_lat * vae.config.scaling_factor
                    gt_lat     = vae.encode(gt_frames).latent_dist.mode()
                    gt_lat     = gt_lat * vae.config.scaling_factor

                input_lat = torch.cat([masked_lat, ref_lat], dim=1).to(weight_dtype)
                timestep  = torch.zeros(input_lat.shape[0], dtype=torch.long,
                                        device=input_lat.device)

                # 音频特征（形状已由 dataset 准备好）
                audio_feat = batch["audio_prompt"].to(weight_dtype)
                if audio_feat.dim() == 5:
                    # [B, T, 10, 5, 384] → [B*T, 50, 384]
                    audio_feat = rearrange(audio_feat, "b t n s d -> (b t) (n s) d")

                # ---------- Teacher 推理（无梯度）----------
                teacher_hook.clear()
                with torch.no_grad():
                    teacher_out = teacher_unet(
                        input_lat, timestep,
                        encoder_hidden_states=audio_feat,
                        return_dict=False,
                    )[0]
                teacher_feats = [teacher_hook.get(l) for l in DISTILL_LAYERS]

                # ---------- Student 推理 ----------
                student_hook.clear()
                student_out = student_unet(
                    input_lat, timestep,
                    encoder_hidden_states=audio_feat,
                    return_dict=False,
                )[0]
                student_feats = [student_hook.get(l) for l in DISTILL_LAYERS]

                # ---------- 损失计算 ----------
                lp = cfg.loss_params

                # 1. 输出蒸馏（最核心）
                L_out = F.mse_loss(student_out, teacher_out.detach())

                # 2. 特征蒸馏（cosine similarity）
                proj_feats = projector(student_feats)
                L_feat = torch.tensor(0.0, device=input_lat.device)
                for pf, tf in zip(proj_feats, teacher_feats):
                    if tf is not None and pf is not None:
                        # 展平空间维度后做 cosine sim
                        pf_flat = pf.flatten(2)              # [B,C,H*W]
                        tf_flat = tf.detach().flatten(2)
                        sim = F.cosine_similarity(pf_flat, tf_flat, dim=1)
                        L_feat = L_feat + (1 - sim.mean())
                L_feat = L_feat / max(len(DISTILL_LAYERS), 1)

                # 3. GT latent L1 重建
                L_recon = F.l1_loss(student_out, gt_lat.detach())

                # 4. VGG 感知损失（在解码图像上）
                L_vgg = torch.tensor(0.0, device=input_lat.device)
                if vgg is not None and lp.vgg_loss > 0:
                    with torch.no_grad():
                        pred_img = vae.decode(
                            student_out / vae.config.scaling_factor
                        ).sample
                        gt_img = vae.decode(
                            gt_lat / vae.config.scaling_factor
                        ).sample
                    L_vgg = vgg(pred_img, gt_img)

                # 5. SyncNet 唇同步损失
                L_sync = torch.tensor(0.0, device=input_lat.device)
                if syncnet is not None and lp.sync_loss > 0:
                    try:
                        L_sync = get_sync_loss(
                            syncnet, batch,
                            student_out / vae.config.scaling_factor,
                            vae, accelerator.device
                        )
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
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---------- 日志 ----------
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "L_out": f"{L_out.item():.4f}",
                    "L_feat": f"{L_feat.item():.4f}",
                    "L_recon": f"{L_recon.item():.4f}",
                })

                if accelerator.is_main_process:
                    accelerator.log({
                        "loss": loss.item(),
                        "L_output_distill": L_out.item(),
                        "L_feat_distill": L_feat.item(),
                        "L_recon": L_recon.item(),
                        "L_vgg": L_vgg.item(),
                        "L_sync": L_sync.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

                # ---------- 保存 ----------
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    student_raw = accelerator.unwrap_model(student_unet)
                    ckpt_path = os.path.join(save_dir, f"student_unet-{global_step}.pth")
                    torch.save(student_raw.state_dict(), ckpt_path)
                    logger.info(f"✓ checkpoint 保存: {ckpt_path}")

                if global_step >= cfg.solver.max_train_steps:
                    break

        if global_step >= cfg.solver.max_train_steps:
            break

    # ==================== 最终保存 ====================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        student_raw = accelerator.unwrap_model(student_unet)
        final_path = os.path.join(save_dir, "student_unet_final.pth")
        torch.save(student_raw.state_dict(), final_path)
        logger.info(f"✓ 训练完成，最终模型: {final_path}")
        logger.info(f"  参数量: {sum(p.numel() for p in student_raw.parameters())/1e6:.1f}M")
        logger.info(f"  FP32 大小估算: {sum(p.numel() for p in student_raw.parameters())*4/1e6:.0f} MB")
        logger.info(f"  INT8 大小估算: {sum(p.numel() for p in student_raw.parameters())*1/1e6:.0f} MB")

    teacher_hook.remove()
    student_hook.remove()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, required=True,
                        help="蒸馏训练配置（distill.yaml）")
    parser.add_argument("--student_config", type=str, default=None,
                        help="Student UNet 架构 JSON（默认 configs/student_musetalk.json）")
    parser.add_argument("--data_root",      type=str, required=True,
                        help="视频数据集根目录（HDTF 或其他）")
    parser.add_argument("--data_list",      type=str, required=True,
                        help="训练数据列表文件（txt）")
    args = parser.parse_args()
    main(args)
