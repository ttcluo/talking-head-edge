"""
P4：ONNX / TensorRT 导出与基准测试

导出三个子图：UNet、VAE Encoder、VAE Decoder
基准顺序：PyTorch FP16 → ONNX Runtime → TensorRT（需 --build_trt）

使用方法：
    conda activate musetalk
    cd ~/MuseTalk

    # 仅导出 ONNX + ORT 基准（约 2 分钟）
    python /path/to/step2/export_onnx.py

    # 额外构建 TensorRT engine（约 5-10 分钟，需要 tensorrt 包）
    python /path/to/step2/export_onnx.py --build_trt

输出：
    - models/onnx/unet.onnx / vae_enc.onnx / vae_dec.onnx
    - （可选）models/trt/unet.trt / vae_enc.trt / vae_dec.trt
    - profile_results/export_onnx.json（各后端 FPS 对比）
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",  type=str, default="profile_results")
parser.add_argument("--onnx_dir",    type=str, default="models/onnx")
parser.add_argument("--trt_dir",     type=str, default="models/trt")
parser.add_argument("--build_trt",   action="store_true", help="构建 TensorRT 引擎")
parser.add_argument("--warmup",      type=int, default=10,  help="预热次数")
parser.add_argument("--bench_runs",  type=int, default=100, help="基准测试次数")
parser.add_argument("--opset",       type=int, default=17,  help="ONNX opset 版本")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.onnx_dir,   exist_ok=True)
os.makedirs(args.trt_dir,    exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  P4：ONNX / TensorRT 导出与基准测试")
print("=" * 65)
print(f"  设备={device}  opset={args.opset}  bench_runs={args.bench_runs}")


# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench_torch(fn, inputs, warmup=10, runs=100):
    """测量 PyTorch 函数平均耗时（ms）"""
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        sync()
        t0 = time.time()
        for _ in range(runs):
            fn(*inputs)
        sync()
    return (time.time() - t0) / runs * 1000


# ==================== 加载模型 ====================
print("\n[1/4] 加载 PyTorch 模型（FP16）")
from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel
import json as _json

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = _json.load(f)

vae  = AutoencoderKL.from_pretrained("models/sd-vae").to(device, torch.float16)
unet = UNet2DConditionModel(**unet_config).to(device, torch.float16)
unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
vae.eval(); unet.eval()
print("  ✓ VAE + UNet 加载完成")


# ==================== 子图封装 ====================
class UNetWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, sample, timestep, enc_hs):
        return self.m(sample, timestep, encoder_hidden_states=enc_hs).sample

class VAEEncWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m.encode(x).latent_dist.mean

class VAEDecWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z): return self.m.decode(z).sample


# ==================== 构造虚拟输入（FP32 供 ONNX 导出，FP16 供 PT 基准）====================
# UNet
unet_in_fp16 = (
    torch.randn(1, 8, 32, 32, device=device, dtype=torch.float16),
    torch.tensor([0.0], device=device, dtype=torch.float16),
    torch.zeros(1, 1, 384, device=device, dtype=torch.float16),
)
unet_in_fp32 = tuple(t.float() for t in unet_in_fp16)

# VAE Encoder
vae_enc_in_fp16 = (torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16),)
vae_enc_in_fp32 = tuple(t.float() for t in vae_enc_in_fp16)

# VAE Decoder
vae_dec_in_fp16 = (torch.randn(1, 4, 32, 32, device=device, dtype=torch.float16),)
vae_dec_in_fp32 = tuple(t.float() for t in vae_dec_in_fp16)


# ==================== PyTorch FP16 基准 ====================
print("\n[2/4] PyTorch FP16 基准")
unet_w   = UNetWrapper(unet).to(device, torch.float16)
vae_enc  = VAEEncWrapper(vae).to(device, torch.float16)
vae_dec  = VAEDecWrapper(vae).to(device, torch.float16)

ms_unet_pt  = bench_torch(unet_w,  unet_in_fp16,    args.warmup, args.bench_runs)
ms_enc_pt   = bench_torch(vae_enc, vae_enc_in_fp16, args.warmup, args.bench_runs)
ms_dec_pt   = bench_torch(vae_dec, vae_dec_in_fp16, args.warmup, args.bench_runs)
ms_total_pt = ms_unet_pt + ms_enc_pt + ms_dec_pt

print(f"  UNet      : {ms_unet_pt:.2f} ms   → {1000/ms_unet_pt:.1f} FPS")
print(f"  VAE Enc   : {ms_enc_pt:.2f} ms")
print(f"  VAE Dec   : {ms_dec_pt:.2f} ms")
print(f"  合计（enc+unet+dec）: {ms_total_pt:.2f} ms → {1000/ms_total_pt:.1f} FPS")


# ==================== ONNX 导出 ====================
print("\n[3/4] ONNX 导出（FP32，opset={})".format(args.opset))

# PyTorch 2.0 的 aten::scaled_dot_product_attention 在 JIT/ONNX 图层面不受
# torch.backends.cuda.enable_flash_sdp 控制，必须在 Python 层 monkey-patch。
import torch.nn.functional as _F

_orig_sdpa = _F.scaled_dot_product_attention

def _onnx_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
               is_causal=False, scale=None):
    """标准 QK^T/√d·V 实现，全部由 ONNX 可表示的算子组成。"""
    scale_factor = (query.size(-1) ** -0.5) if scale is None else scale
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn = attn.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = torch.dropout(attn, dropout_p, train=True)
    return torch.matmul(attn, value)

_F.scaled_dot_product_attention = _onnx_sdpa
print("  ✓ F.scaled_dot_product_attention 已替换为 ONNX 兼容实现")

# 导出需要 FP32（ONNX 导出时 FP16 算子支持不稳定）
unet_fp32  = UNetWrapper(unet.float()).eval()
enc_fp32   = VAEEncWrapper(vae.float()).eval()
dec_fp32   = VAEDecWrapper(vae.float()).eval()

onnx_unet = os.path.join(args.onnx_dir, "unet.onnx")
onnx_enc  = os.path.join(args.onnx_dir, "vae_enc.onnx")
onnx_dec  = os.path.join(args.onnx_dir, "vae_dec.onnx")

def export(model, inputs, path, input_names, output_names, dynamic_axes):
    if os.path.exists(path):
        print(f"  ↻ 已存在，跳过导出: {path}")
        return
    with torch.no_grad():
        torch.onnx.export(
            model, inputs, path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ {os.path.basename(path):20s} {size_mb:.1f} MB")

print("  导出 UNet ...")
export(
    unet_fp32, unet_in_fp32, onnx_unet,
    input_names=["sample", "timestep", "enc_hs"],
    output_names=["out_sample"],
    dynamic_axes={"sample": {0: "B"}, "enc_hs": {0: "B"}, "out_sample": {0: "B"}},
)

print("  导出 VAE Encoder ...")
export(
    enc_fp32, vae_enc_in_fp32, onnx_enc,
    input_names=["image"],
    output_names=["latent"],
    dynamic_axes={"image": {0: "B"}, "latent": {0: "B"}},
)

print("  导出 VAE Decoder ...")
export(
    dec_fp32, vae_dec_in_fp32, onnx_dec,
    input_names=["latent"],
    output_names=["image"],
    dynamic_axes={"latent": {0: "B"}, "image": {0: "B"}},
)

# 恢复原始 SDPA，切回 FP16 用于后续推理
_F.scaled_dot_product_attention = _orig_sdpa
unet.half(); vae.half()


# ==================== ONNX Runtime 基准 ====================
print("\n  ONNX Runtime 基准")
try:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device == "cuda" else ["CPUExecutionProvider"]

    sess_unet = ort.InferenceSession(onnx_unet, providers=providers)
    sess_enc  = ort.InferenceSession(onnx_enc,  providers=providers)
    sess_dec  = ort.InferenceSession(onnx_dec,  providers=providers)

    # 转为 CPU numpy（ORT 接受 numpy）
    def to_np(*ts): return [t.cpu().float().numpy() for t in ts]

    ui = to_np(*unet_in_fp32)
    ei = to_np(*vae_enc_in_fp32)
    di = to_np(*vae_dec_in_fp32)

    def run_unet_ort():
        sess_unet.run(None, {"sample": ui[0], "timestep": ui[1], "enc_hs": ui[2]})
    def run_enc_ort():
        sess_enc.run(None, {"image": ei[0]})
    def run_dec_ort():
        sess_dec.run(None, {"latent": di[0]})

    # 预热
    for _ in range(args.warmup):
        run_unet_ort(); run_enc_ort(); run_dec_ort()

    t0 = time.time()
    for _ in range(args.bench_runs): run_unet_ort()
    ms_unet_ort = (time.time() - t0) / args.bench_runs * 1000

    t0 = time.time()
    for _ in range(args.bench_runs): run_enc_ort()
    ms_enc_ort = (time.time() - t0) / args.bench_runs * 1000

    t0 = time.time()
    for _ in range(args.bench_runs): run_dec_ort()
    ms_dec_ort = (time.time() - t0) / args.bench_runs * 1000

    ms_total_ort = ms_unet_ort + ms_enc_ort + ms_dec_ort
    ort_ok = True
    print(f"  UNet      : {ms_unet_ort:.2f} ms   → {1000/ms_unet_ort:.1f} FPS  (vs PT {ms_unet_pt:.2f}ms, {ms_unet_pt/ms_unet_ort:.2f}×)")
    print(f"  VAE Enc   : {ms_enc_ort:.2f} ms")
    print(f"  VAE Dec   : {ms_dec_ort:.2f} ms")
    print(f"  合计: {ms_total_ort:.2f} ms → {1000/ms_total_ort:.1f} FPS  (vs PT {ms_total_pt/ms_total_ort:.2f}×)")

except ImportError:
    print("  ⚠ onnxruntime 未安装（pip install onnxruntime-gpu），跳过 ORT 基准")
    ort_ok = False
    ms_unet_ort = ms_enc_ort = ms_dec_ort = ms_total_ort = None


# ==================== TensorRT 构建与基准 ====================
trt_ok = False
ms_unet_trt = ms_enc_trt = ms_dec_trt = ms_total_trt = None

if args.build_trt:
    print("\n  TensorRT 引擎构建（此过程约 5-10 分钟）")
    try:
        import tensorrt as trt_lib
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        TRT_LOGGER = trt_lib.Logger(trt_lib.Logger.WARNING)

        def build_engine(onnx_path, trt_path, fp16=True):
            if os.path.exists(trt_path):
                print(f"  ↻ 已存在，加载 engine: {trt_path}")
                with open(trt_path, "rb") as f, \
                     trt_lib.Runtime(TRT_LOGGER) as rt:
                    return rt.deserialize_cuda_engine(f.read())

            builder = trt_lib.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt_lib.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt_lib.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"  ✗ ONNX parse error: {parser.get_error(i)}")
                    return None

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt_lib.MemoryPoolType.WORKSPACE, 4 << 30)
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt_lib.BuilderFlag.FP16)

            engine = builder.build_serialized_network(network, config)
            if engine is None:
                return None
            with open(trt_path, "wb") as f:
                f.write(engine)
            size_mb = os.path.getsize(trt_path) / 1024 / 1024
            print(f"  ✓ {os.path.basename(trt_path):20s} {size_mb:.1f} MB")
            with trt_lib.Runtime(TRT_LOGGER) as rt:
                return rt.deserialize_cuda_engine(engine)

        def trt_infer(engine, input_dict):
            context = engine.create_execution_context()
            bindings = []
            outputs = {}
            for i in range(engine.num_io_tensors):
                name  = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = trt_lib.nptype(engine.get_tensor_dtype(name))
                buf   = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
                bindings.append(int(buf))
                if engine.get_tensor_mode(name) == trt_lib.TensorIOMode.INPUT:
                    cuda.memcpy_htod(buf, input_dict[name].astype(dtype))
                else:
                    outputs[name] = (buf, shape, dtype)
                context.set_tensor_address(name, int(buf))
            context.execute_v2(bindings)
            results = {}
            for name, (buf, shape, dtype) in outputs.items():
                arr = np.empty(shape, dtype=dtype)
                cuda.memcpy_dtoh(arr, buf)
                results[name] = arr
            return results

        trt_unet_path = os.path.join(args.trt_dir, "unet.trt")
        trt_enc_path  = os.path.join(args.trt_dir, "vae_enc.trt")
        trt_dec_path  = os.path.join(args.trt_dir, "vae_dec.trt")

        print("  构建 UNet engine ...")
        eng_unet = build_engine(onnx_unet, trt_unet_path)
        print("  构建 VAE Encoder engine ...")
        eng_enc  = build_engine(onnx_enc,  trt_enc_path)
        print("  构建 VAE Decoder engine ...")
        eng_dec  = build_engine(onnx_dec,  trt_dec_path)

        if eng_unet and eng_enc and eng_dec:
            ui_np = {k: v for k, v in zip(
                ["sample", "timestep", "enc_hs"],
                [t.cpu().float().numpy() for t in unet_in_fp32])}
            ei_np = {"image":   vae_enc_in_fp32[0].cpu().float().numpy()}
            di_np = {"latent":  vae_dec_in_fp32[0].cpu().float().numpy()}

            for _ in range(args.warmup):
                trt_infer(eng_unet, ui_np)
                trt_infer(eng_enc,  ei_np)
                trt_infer(eng_dec,  di_np)

            t0 = time.time()
            for _ in range(args.bench_runs): trt_infer(eng_unet, ui_np)
            ms_unet_trt = (time.time() - t0) / args.bench_runs * 1000

            t0 = time.time()
            for _ in range(args.bench_runs): trt_infer(eng_enc, ei_np)
            ms_enc_trt = (time.time() - t0) / args.bench_runs * 1000

            t0 = time.time()
            for _ in range(args.bench_runs): trt_infer(eng_dec, di_np)
            ms_dec_trt = (time.time() - t0) / args.bench_runs * 1000

            ms_total_trt = ms_unet_trt + ms_enc_trt + ms_dec_trt
            trt_ok = True
            print(f"\n  TensorRT 结果：")
            print(f"  UNet    : {ms_unet_trt:.2f} ms  (vs PT {ms_unet_pt/ms_unet_trt:.2f}×)")
            print(f"  VAE Enc : {ms_enc_trt:.2f} ms")
            print(f"  VAE Dec : {ms_dec_trt:.2f} ms")
            print(f"  合计    : {ms_total_trt:.2f} ms → {1000/ms_total_trt:.1f} FPS  (vs PT {ms_total_pt/ms_total_trt:.2f}×)")

    except ImportError as e:
        print(f"  ⚠ TensorRT 未安装（{e}），跳过 TRT 基准")
    except Exception as e:
        print(f"  ✗ TensorRT 构建失败（{e}）")


# ==================== 联合加速比估算 ====================
# P3+ 实测：基线 22.3 FPS，MATS 45.6 FPS（2.05×），跳过率 58%
MATS_SKIP_RATE = 0.58
BASELINE_FPS   = 22.3
MATS_SPEEDUP   = 2.05
MATS_FPS       = BASELINE_FPS * MATS_SPEEDUP

print("\n" + "=" * 70)
print("  P4 汇总：各后端 UNet 耗时与联合加速估算")
print("=" * 70)

rows = [f"  {'PyTorch FP16':<22} {ms_unet_pt:>8.2f}  {ms_total_pt:>8.2f}  {1000/ms_total_pt:>10.1f}   1.00×"]
if ort_ok:
    rows.append(f"  {'ONNX Runtime(全FP32)':<22} {ms_unet_ort:>8.2f}  {ms_total_ort:>8.2f}  {1000/ms_total_ort:>10.1f}   {ms_total_pt/ms_total_ort:.2f}×")
if trt_ok:
    rows.append(f"  {'TensorRT FP16':<22} {ms_unet_trt:>8.2f}  {ms_total_trt:>8.2f}  {1000/ms_total_trt:>10.1f}   {ms_total_pt/ms_total_trt:.2f}×")

print(f"\n  {'后端':<22} {'UNet(ms)':>8}  {'合计(ms)':>8}  {'全管线FPS':>10}   vs基线")
print(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*10}   {'-'*6}")
for r in rows:
    print(r)

# 混合部署场景：ORT/TRT UNet + PT FP16 VAE（实际部署最优策略）
# 跳过帧仅需 PT VAE Enc（运动检测），非跳过帧需 UNet + PT VAE enc + dec
print(f"\n  【关键指标】MATS + 混合部署（ORT/TRT UNet + PT FP16 VAE）")
if ort_ok:
    # 非跳过帧: ORT UNet + PT enc + PT dec
    ms_full_hybrid_ort  = ms_unet_ort + ms_enc_pt + ms_dec_pt
    # 跳过帧: 只需 PT enc 做运动检测，直接复用像素帧
    ms_skip_hybrid      = ms_enc_pt
    ms_avg_hybrid_ort   = (1 - MATS_SKIP_RATE) * ms_full_hybrid_ort + MATS_SKIP_RATE * ms_skip_hybrid
    fps_hybrid_ort      = 1000 / ms_avg_hybrid_ort
    print(f"  MATS + ORT UNet 混合：{ms_avg_hybrid_ort:.1f}ms/帧 → {fps_hybrid_ort:.1f} FPS  ({fps_hybrid_ort/BASELINE_FPS:.2f}× vs 基线)")
if trt_ok:
    ms_full_hybrid_trt  = ms_unet_trt + ms_enc_pt + ms_dec_pt
    ms_avg_hybrid_trt   = (1 - MATS_SKIP_RATE) * ms_full_hybrid_trt + MATS_SKIP_RATE * ms_enc_pt
    fps_hybrid_trt      = 1000 / ms_avg_hybrid_trt
    print(f"  MATS + TRT UNet 混合：{ms_avg_hybrid_trt:.1f}ms/帧 → {fps_hybrid_trt:.1f} FPS  ({fps_hybrid_trt/BASELINE_FPS:.2f}× vs 基线)")

print(f"\n  注：ORT 全 FP32 导致 VAE 变慢，混合方案（ORT/TRT UNet + PT FP16 VAE）才是真实部署场景")
print()

# 未运行的分支补默认值，供 JSON 写入使用
if not ort_ok:
    fps_hybrid_ort = None
if not trt_ok:
    fps_hybrid_trt = None

# ==================== 保存 ====================
result = {
    "config": {"warmup": args.warmup, "bench_runs": args.bench_runs, "opset": args.opset},
    "pytorch_fp16": {
        "unet_ms": round(ms_unet_pt, 2),
        "vae_enc_ms": round(ms_enc_pt, 2),
        "vae_dec_ms": round(ms_dec_pt, 2),
        "total_ms": round(ms_total_pt, 2),
        "fps": round(1000 / ms_total_pt, 1),
    },
    "onnx_runtime": {
        "unet_ms": round(ms_unet_ort, 2) if ort_ok else None,
        "vae_enc_ms": round(ms_enc_ort, 2) if ort_ok else None,
        "vae_dec_ms": round(ms_dec_ort, 2) if ort_ok else None,
        "total_ms": round(ms_total_ort, 2) if ort_ok else None,
        "fps": round(1000 / ms_total_ort, 1) if ort_ok else None,
        "speedup_vs_pt": round(ms_total_pt / ms_total_ort, 2) if ort_ok else None,
    },
    "tensorrt": {
        "unet_ms": round(ms_unet_trt, 2) if trt_ok else None,
        "vae_enc_ms": round(ms_enc_trt, 2) if trt_ok else None,
        "vae_dec_ms": round(ms_dec_trt, 2) if trt_ok else None,
        "total_ms": round(ms_total_trt, 2) if trt_ok else None,
        "fps": round(1000 / ms_total_trt, 1) if trt_ok else None,
        "speedup_vs_pt": round(ms_total_pt / ms_total_trt, 2) if trt_ok else None,
    },
    "mats_context": {
        "baseline_fps": BASELINE_FPS,
        "mats_fps": MATS_FPS,
        "mats_speedup": MATS_SPEEDUP,
        "mats_skip_rate": MATS_SKIP_RATE,
    },
    "hybrid_deployment": {
        "description": "MATS 像素缓存 + ORT/TRT UNet + PT FP16 VAE（实际部署最优策略）",
        "mats_ort_unet_fps": round(fps_hybrid_ort, 1) if ort_ok else None,
        "mats_ort_unet_speedup": round(fps_hybrid_ort / BASELINE_FPS, 2) if ort_ok else None,
        "mats_trt_unet_fps": round(fps_hybrid_trt, 1) if trt_ok else None,
        "mats_trt_unet_speedup": round(fps_hybrid_trt / BASELINE_FPS, 2) if trt_ok else None,
    },
    "onnx_files": {
        "unet": onnx_unet, "vae_enc": onnx_enc, "vae_dec": onnx_dec,
    },
}
out = os.path.join(args.output_dir, "export_onnx.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  结果已保存: {out}")
