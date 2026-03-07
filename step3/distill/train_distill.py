"""
MuseTalk UNet 知识蒸馏训练脚本（轻量版）。

不依赖 mmpose/decord/HDTF，直接使用 realtime_inference.py 预处理产出的
avatar latents + Whisper 音频特征。

Teacher: musetalkV15/unet.pth  (~850MB FP32, [320,640,1280,1280])
Student: student_musetalk.json (~136MB FP32, [128,256,512,512])

蒸馏损失：
  L_total = λ1 * L_output  (MSE student_out vs teacher_out)
          + λ2 * L_feat    (cosine_sim 中间特征对齐)

运行方式（在 $MUSE_ROOT 目录下）：
  cd $MUSE_ROOT && git pull origin main
  PYTHONPATH=$MUSE_ROOT accelerate launch --num_processes 4 \\
      $REPO/step3/distill/train_distill.py \\
      --config    $REPO/step3/distill/configs/distill.yaml \\
      --student_config $REPO/step3/distill/configs/student_musetalk.json \\
      --avatar_list dataset/distill/train_avatars.txt
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.models.unet import PositionalEncoding
from musetalk.utils.utils import load_all_model

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


# ==================== 特征 Hook ====================

class FeatureHook:
    def __init__(self):
        self._feats: dict = {}
        self._handles = []

    def register(self, module: nn.Module, name: str):
        def hook(_, __, output):
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
    def __init__(self, student_ch: list, teacher_ch: list):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(s, t, 1, bias=False)
            for s, t in zip(student_ch, teacher_ch)
        ])

    def forward(self, feats: list) -> list:
        return [proj(f) for proj, f in zip(self.projs, feats)]


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

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        cfg.solver.mixed_precision, torch.float32
    )

    # ==================== Teacher（冻结）====================
    logger.info("加载 Teacher UNet...")
    vae, teacher_wrapper, pe = load_all_model(
        unet_model_path=cfg.teacher_unet_path,
        unet_config=cfg.teacher_unet_config,
        device="cpu",
    )
    teacher_unet = teacher_wrapper.model
    teacher_unet.set_attn_processor(AttnProcessor())
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()
    vae.vae.requires_grad_(False)
    logger.info(f"Teacher 参数量: {sum(p.numel() for p in teacher_unet.parameters())/1e6:.1f}M")

    # ==================== Student（训练）====================
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

    # ==================== 特征 Hook ====================
    LAYERS = list(cfg.feat_distill_layers)
    teacher_hook = FeatureHook()
    student_hook = FeatureHook()

    def _get_module(unet, path):
        m = unet
        for p in path.split("."):
            m = getattr(m, p)
        return m

    for ln in LAYERS:
        teacher_hook.register(_get_module(teacher_unet, ln), ln)
        student_hook.register(_get_module(student_unet, ln), ln)

    teacher_ch = [320, 640, 1280, 640, 320]
    student_ch  = [128, 256,  512, 256, 128]
    projector = FeatProjector(student_ch, teacher_ch)

    # ==================== 数据集 ====================
    sys.path.insert(0, os.path.dirname(__file__))
    from avatar_dataset import build_distill_dataloaders

    avatar_list = args.avatar_list or "dataset/distill/train_avatars.txt"
    train_dl, val_dl = build_distill_dataloaders(
        avatar_list_file=avatar_list,
        avatar_base=cfg.get("avatar_base", "results/v15/avatars"),
        audio_feat_dir=cfg.get("audio_feat_dir", "dataset/distill/audio_feats"),
        batch_size=cfg.data.train_bs,
        num_workers=cfg.data.num_workers,
        samples_per_avatar=cfg.get("samples_per_avatar", 500),
    )
    logger.info(f"训练样本: {len(train_dl.dataset)}  验证样本: {len(val_dl.dataset)}")

    # ==================== 优化器 ====================
    trainable = list(student_unet.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
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
        student_unet, projector, teacher_unet, vae.vae,
        optimizer, train_dl, lr_scheduler,
    ) = accelerator.prepare(
        student_unet, projector, teacher_unet, vae.vae,
        optimizer, train_dl, lr_scheduler,
    )
    # 冻结模型转 FP16 节省显存，可训练参数必须保持 FP32（AMP 要求）
    teacher_unet.to(weight_dtype)
    vae.vae.to(weight_dtype)
    pe = pe.to(accelerator.device)

    lp = cfg.loss_params
    global_step = 0
    progress_bar = tqdm(
        range(cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    logger.info("=" * 60)
    logger.info("  开始蒸馏训练")
    logger.info("=" * 60)

    for _ in range(99999):
        student_unet.train()
        projector.train()

        for batch in train_dl:
            with accelerator.accumulate(student_unet):

                latent   = batch["latent"].to(accelerator.device)      # [B, 8, 32, 32] FP32
                audio_f  = batch["audio_feat"].to(accelerator.device)  # [B, 50, 384]   FP32
                timestep = torch.zeros(
                    latent.shape[0], dtype=torch.long, device=latent.device
                )

                # 音频位置编码（与推理一致）
                audio_f = pe(audio_f)  # [B, 50, 384]

                # ---------- Teacher（FP16）----------
                teacher_hook.clear()
                with torch.no_grad():
                    teacher_out = teacher_unet(
                        latent.to(weight_dtype), timestep,
                        encoder_hidden_states=audio_f.to(weight_dtype),
                        return_dict=False,
                    )[0].float()
                t_feats = [f.float() if f is not None else None for f in [teacher_hook.get(l) for l in LAYERS]]

                # ---------- Student（FP32，AMP autocast 由 accelerator 处理）----------
                student_hook.clear()
                student_out = student_unet(
                    latent, timestep,
                    encoder_hidden_states=audio_f,
                    return_dict=False,
                )[0]
                s_feats = [student_hook.get(l) for l in LAYERS]

                # ---------- 损失 ----------
                L_out  = F.mse_loss(student_out, teacher_out.detach())

                proj_feats = projector(s_feats)
                L_feat = torch.tensor(0.0, device=latent.device)
                valid  = 0
                for pf, tf in zip(proj_feats, t_feats):
                    if tf is not None and pf is not None:
                        sim = F.cosine_similarity(
                            pf.flatten(2), tf.detach().flatten(2), dim=1
                        )
                        L_feat = L_feat + (1 - sim.mean())
                        valid += 1
                if valid:
                    L_feat = L_feat / valid

                loss = lp.output_distill * L_out + lp.feat_distill * L_feat

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss":   f"{loss.item():.4f}",
                    "L_out":  f"{L_out.item():.4f}",
                    "L_feat": f"{L_feat.item():.4f}",
                })

                if accelerator.is_main_process:
                    accelerator.log({
                        "loss":   loss.item(),
                        "L_out":  L_out.item(),
                        "L_feat": L_feat.item(),
                        "lr":     lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    raw = accelerator.unwrap_model(student_unet)
                    ckpt = os.path.join(save_dir, f"student_unet-{global_step}.pth")
                    torch.save(raw.state_dict(), ckpt)
                    logger.info(f"✓ checkpoint: {ckpt}")

                if global_step >= cfg.solver.max_train_steps:
                    break

        if global_step >= cfg.solver.max_train_steps:
            break

    # ==================== 保存 ====================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(student_unet)
        final = os.path.join(save_dir, "student_unet_final.pth")
        torch.save(raw.state_dict(), final)
        n = sum(p.numel() for p in raw.parameters())
        logger.info(f"✓ 训练完成: {final}")
        logger.info(f"  参数量: {n/1e6:.1f}M  FP32≈{n*4/1e6:.0f}MB  INT8≈{n/1e6:.0f}MB")

    teacher_hook.remove()
    student_hook.remove()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--student_config", type=str, default=None)
    parser.add_argument("--avatar_list",    type=str, default=None,
                        help="avatar 列表文件（默认 dataset/distill/train_avatars.txt）")
    args = parser.parse_args()
    main(args)
