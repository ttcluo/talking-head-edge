"""
轻量蒸馏数据集：直接读取 MuseTalk avatar 预处理产物，不依赖 mmpose/decord。

Avatar 预处理输出（来自 realtime_inference.py）：
  results/v15/avatars/<avatar_id>/
    full_imgs/          全帧图像
    coords.pkl          人脸坐标列表
    latents/
      unet_input_latent_list.pt   已预处理的 UNet 输入 latent 列表

每个 latent 形状：[1, 8, 32, 32]（4ch masked + 4ch ref）
"""

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from musetalk.whisper.audio2feature import Audio2Feature


class AvatarDistillDataset(Dataset):
    """从 avatar 预处理结果构建蒸馏训练数据集。

    每个样本返回：
        latent      [8, 32, 32]  UNet 输入（masked_latent cat ref_latent）
        audio_feat  [50, 384]    Whisper 音频特征（pe 之前）
    """

    def __init__(
        self,
        avatar_ids: list,
        avatar_base: str = "results/v15/avatars",
        audio_dir: str = "data/audio",
        whisper_model_path: str = "models/whisper/tiny.pt",
        fps: int = 25,
        samples_per_avatar: int = 200,
    ):
        self.samples = []  # [(latent_tensor, audio_tensor), ...]

        if not os.path.isabs(whisper_model_path):
            muse_root = os.environ.get("MUSE_ROOT", os.getcwd())
            whisper_model_path = os.path.join(muse_root, whisper_model_path)
        audio2feat = Audio2Feature(model_path=whisper_model_path)

        for avatar_id in avatar_ids:
            avatar_dir = os.path.join(avatar_base, avatar_id)
            latent_path = os.path.join(avatar_dir, "latents", "unet_input_latent_list.pt")
            if not os.path.exists(latent_path):
                print(f"  ⚠ {avatar_id}: latent 文件不存在，跳过")
                continue

            latent_list = torch.load(latent_path, map_location="cpu")  # list of [1,8,32,32]
            if not latent_list:
                continue

            # 找匹配的音频文件
            vname = avatar_id.replace("avator_", "")
            audio_path = os.path.join(audio_dir, f"{vname}.wav")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(audio_dir, "yongen.wav")
            if not os.path.exists(audio_path):
                print(f"  ⚠ {avatar_id}: 找不到音频，跳过")
                continue

            # 提取 Whisper 特征块
            try:
                whisper_feat = audio2feat.get_hubert_from_whisper(audio_path)
                chunks = audio2feat.feature2chunks(feature_array=whisper_feat, fps=fps)
            except Exception as e:
                print(f"  ⚠ {avatar_id}: 音频特征提取失败 ({e})，跳过")
                continue

            n_latents = len(latent_list)
            n_chunks  = len(chunks)
            n_samples = min(samples_per_avatar, n_latents, n_chunks)

            for i in range(n_samples):
                lat = latent_list[i % n_latents]  # [1, 8, 32, 32]
                if isinstance(lat, torch.Tensor):
                    lat = lat.squeeze(0)           # [8, 32, 32]
                else:
                    lat = torch.tensor(lat).squeeze(0)

                af = chunks[i % n_chunks]
                if isinstance(af, torch.Tensor):
                    af = af.float()
                else:
                    af = torch.tensor(af, dtype=torch.float32)  # [50, 384]

                self.samples.append((lat, af))

            print(f"  ✓ {avatar_id}: {n_samples} 样本（latents={n_latents}, chunks={n_chunks}）")

        print(f"  数据集总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lat, af = self.samples[idx]
        return {
            "latent":     lat.float(),     # [8, 32, 32]
            "audio_feat": af.float(),      # [50, 384]
        }


def build_distill_dataloaders(
    avatar_list_file: str,
    avatar_base: str,
    audio_dir: str,
    whisper_model_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    samples_per_avatar: int = 500,
    val_split: float = 0.1,
):
    """从 avatar 列表文件构建 train/val DataLoader。"""
    from torch.utils.data import DataLoader, random_split

    with open(avatar_list_file) as f:
        avatar_ids = [l.strip() for l in f if l.strip()]

    dataset = AvatarDistillDataset(
        avatar_ids=avatar_ids,
        avatar_base=avatar_base,
        audio_dir=audio_dir,
        whisper_model_path=whisper_model_path,
        samples_per_avatar=samples_per_avatar,
    )

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=1,           pin_memory=True)

    return train_dl, val_dl
