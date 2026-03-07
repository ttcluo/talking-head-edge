"""
轻量蒸馏数据集：直接读取 MuseTalk avatar 预处理产物，不依赖 mmpose/decord。

音频特征需提前用 precompute_audio_feats.py 离线生成：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/precompute_audio_feats.py \\
      --avatar_list dataset/distill/train_avatars.txt

每个样本返回：
    latent      [8, 32, 32]  UNet 输入（masked_latent cat ref_latent）
    audio_feat  [50, 384]    Whisper 音频特征
"""

import os

import torch
from torch.utils.data import Dataset


class AvatarDistillDataset(Dataset):
    def __init__(
        self,
        avatar_ids: list,
        avatar_base: str = "results/v15/avatars",
        audio_feat_dir: str = "dataset/distill/audio_feats",
        samples_per_avatar: int = 500,
    ):
        self.samples = []

        for avatar_id in avatar_ids:
            latent_path = os.path.join(
                avatar_base, avatar_id, "latents", "unet_input_latent_list.pt"
            )
            audio_path = os.path.join(audio_feat_dir, f"{avatar_id}.pt")

            if not os.path.exists(latent_path):
                print(f"  ⚠ {avatar_id}: latent 文件不存在，跳过")
                continue
            if not os.path.exists(audio_path):
                print(f"  ⚠ {avatar_id}: 音频特征不存在，请先运行 precompute_audio_feats.py，跳过")
                continue

            latent_list = torch.load(latent_path, map_location="cpu")
            audio_chunks = torch.load(audio_path, map_location="cpu")

            if not latent_list or not audio_chunks:
                continue

            n = min(samples_per_avatar, len(latent_list), len(audio_chunks))
            for i in range(n):
                lat = latent_list[i]
                if isinstance(lat, torch.Tensor):
                    lat = lat.squeeze(0)
                else:
                    lat = torch.tensor(lat).squeeze(0)

                af = audio_chunks[i]
                if not isinstance(af, torch.Tensor):
                    af = torch.tensor(af, dtype=torch.float32)

                self.samples.append((lat.float(), af.float()))

            print(f"  ✓ {avatar_id}: {n} 样本（latents={len(latent_list)}, audio={len(audio_chunks)}）")

        print(f"  数据集总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lat, af = self.samples[idx]
        return {"latent": lat, "audio_feat": af}


def build_distill_dataloaders(
    avatar_list_file: str,
    avatar_base: str,
    audio_feat_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    samples_per_avatar: int = 500,
    val_split: float = 0.1,
):
    from torch.utils.data import DataLoader, random_split

    with open(avatar_list_file) as f:
        avatar_ids = [l.strip() for l in f if l.strip()]

    dataset = AvatarDistillDataset(
        avatar_ids=avatar_ids,
        avatar_base=avatar_base,
        audio_feat_dir=audio_feat_dir,
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
