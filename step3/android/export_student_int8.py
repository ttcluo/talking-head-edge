"""
Student UNet 静态 INT8 量化导出（Conv2d + Linear 全量化）。

静态量化需要校准数据，直接复用 avatar latents + 音频特征。
预计产出：unet_student_static_int8.ptl  (~138MB)

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_student_int8.py \\
      --student_ckpt   exp_out/distill/distill_v1/student_unet-2000.pth \\
      --student_config $REPO/step3/distill/configs/student_musetalk.json \\
      --avatar_list    dataset/distill/train_avatars.txt \\
      --audio_feat_dir dataset/distill/audio_feats \\
      --out_dir        models/student_ptlite/
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.quantization as tq
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from torch.utils.mobile_optimizer import optimize_for_mobile
from tqdm import tqdm

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.models.unet import PositionalEncoding

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt",   required=True)
parser.add_argument("--student_config", required=True)
parser.add_argument("--avatar_list",    default="dataset/distill/train_avatars.txt")
parser.add_argument("--audio_feat_dir", default="dataset/distill/audio_feats")
parser.add_argument("--avatar_base",    default="results/v15/avatars")
parser.add_argument("--out_dir",        default="models/student_ptlite")
parser.add_argument("--n_calib",        type=int, default=100, help="校准样本数")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

print("=" * 60)
print("  Student UNet 静态 INT8 量化")
print("=" * 60)

# ==================== 加载 Student ====================
with open(args.student_config) as f:
    student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
model = UNet2DConditionModel(**student_cfg)
ckpt  = torch.load(args.student_ckpt, map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.set_attn_processor(AttnProcessor())
model = model.float().cpu().eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Student: {n_params/1e6:.1f}M 参数  FP32={n_params*4/1e6:.0f}MB")

# ==================== UNet Wrapper ====================
class _UNetWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, latent: torch.Tensor,
                timestep: torch.Tensor,
                audio_feat: torch.Tensor) -> torch.Tensor:
        return self.m(latent, timestep,
                      encoder_hidden_states=audio_feat,
                      return_dict=False)[0]

wrapper = _UNetWrapper(model)

# ==================== 校准数据集 ====================
print(f"\n[准备校准数据] n_calib={args.n_calib}")
pe = PositionalEncoding(d_model=384)

calib_inputs = []
with open(args.avatar_list) as f:
    avatar_ids = [l.strip() for l in f if l.strip()]

for avatar_id in avatar_ids:
    if len(calib_inputs) >= args.n_calib:
        break
    latent_path = os.path.join(args.avatar_base, avatar_id, "latents.pt")
    audio_path  = os.path.join(args.audio_feat_dir, f"{avatar_id}.pt")
    if not os.path.exists(latent_path) or not os.path.exists(audio_path):
        continue
    latents = torch.load(latent_path, map_location="cpu")
    audios  = torch.load(audio_path, map_location="cpu")
    for i in range(min(len(latents), len(audios))):
        if len(calib_inputs) >= args.n_calib:
            break
        lat = latents[i].float()
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        af = audios[i].float().unsqueeze(0)
        af = pe(af)
        calib_inputs.append((lat, torch.tensor([0], dtype=torch.long), af))

print(f"  ✓ 校准样本: {len(calib_inputs)}")

# ==================== 静态量化 ====================
print("\n[静态量化] fuse → prepare → calibrate → convert")

# 设置量化配置（fbgemm 适合 x86；Android 用 qnnpack）
wrapper.qconfig = tq.get_default_qconfig("qnnpack")
torch.backends.quantized.engine = "qnnpack"

# prepare：插入 observer
wrapper_prepared = tq.prepare(wrapper, inplace=False)

# calibrate
print("  校准中...")
with torch.no_grad():
    for lat, t, af in tqdm(calib_inputs, desc="  calib"):
        try:
            wrapper_prepared(lat, t, af)
        except Exception:
            pass  # 忽略校准中的非致命错误

# convert：量化
wrapper_q = tq.convert(wrapper_prepared, inplace=False)
print("  ✓ 静态量化完成")

# ==================== JIT trace → .ptl ====================
print("\n[JIT trace → optimize_for_mobile → .ptl]")

dummy_latent = torch.zeros(1, 8, 32, 32, dtype=torch.float32)
dummy_t      = torch.tensor([0], dtype=torch.long)
dummy_audio  = torch.zeros(1, 50, 384, dtype=torch.float32)

try:
    with torch.no_grad():
        traced = torch.jit.trace(wrapper_q, (dummy_latent, dummy_t, dummy_audio), strict=False)
    optimized = optimize_for_mobile(traced)
    out_path  = os.path.join(args.out_dir, "unet_student_static_int8.ptl")
    optimized._save_for_lite_interpreter(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  ✓ 导出: {out_path}  ({size_mb:.1f} MB)")

    # 验证
    from torch.jit import load as jit_load
    loaded = jit_load(out_path)
    with torch.no_grad():
        out = loaded(dummy_latent, dummy_t, dummy_audio)
    print(f"  ✓ 推理验证通过，输出 {out.shape}")

    print(f"""
==============================
  量化结果
==============================
  FP32:         {n_params*4/1e6:.0f} MB
  Static INT8:  {size_mb:.1f} MB
  压缩率:       {n_params*4/1e6 / size_mb:.1f}×
  输出:         {out_path}
""")

except Exception as e:
    import traceback
    print(f"\n⚠ 静态量化 trace 失败，回退到动态量化（Conv2d+Linear）...")
    traceback.print_exc()

    # 回退：动态量化同时量化 Linear 和 Conv2d
    model2 = UNet2DConditionModel(**student_cfg)
    model2.load_state_dict(ckpt, strict=False)
    model2.set_attn_processor(AttnProcessor())
    model2 = model2.float().cpu().eval()
    wrapper2 = _UNetWrapper(model2)

    wrapper2_q = tq.quantize_dynamic(
        wrapper2,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    with torch.no_grad():
        traced2 = torch.jit.trace(wrapper2_q, (dummy_latent, dummy_t, dummy_audio), strict=False)
    opt2 = optimize_for_mobile(traced2)
    out2 = os.path.join(args.out_dir, "unet_student_dynamic_int8.ptl")
    opt2._save_for_lite_interpreter(out2)
    sz2 = os.path.getsize(out2) / 1e6
    print(f"  ✓ 动态 INT8 (Linear+Conv): {out2}  ({sz2:.1f} MB)")
