# 蒸馏训练数据集准备

蒸馏训练需要两类数据（均在 **MuseTalk 项目根目录 `$MUSE_ROOT`** 下）：

1. **Avatar 预处理**：每个说话人对应 `results/v15/avatars/<avatar_id>/`，内含 `latents.pt`（UNet 输入 latent 序列）。
2. **音频特征**：`dataset/distill/audio_feats/<avatar_id>.pt`，由 `precompute_audio_feats.py` 从对应音频生成。

下面按「下载 HDTF → 生成 avatar → 生成音频特征 → 写 train_avatars.txt」顺序说明。

---

## 一、下载 HDTF（或自备说话人视频）

### 1.1 HDTF 数据集（可访问 Hugging Face 时）

- **简介**：High-Definition Talking Face，说话人视频，MuseTalk 官方训练使用。
- **获取方式**：
  - Hugging Face：<https://huggingface.co/datasets/global-optima-research/HDTF>（按页说明下载）。
  - 或从论文/官方仓库获取链接：<https://github.com/MRzzm/HDTF>。
- **放置**：将下载得到的 **MP4 视频** 放到统一目录，例如 `$MUSE_ROOT/dataset/HDTF/source/`。若使用 MuseTalk 自带预处理，需保证 `configs/training/preprocess.yaml` 里 `video_root_raw` 指向该目录。
- **若下载到的是 videos.zip**：放在 `dataset/HDTF/source/` 下后，用本仓库脚本解压并整理为同一目录 MP4：
  ```bash
  python $REPO/step3/distill/unzip_hdtf_videos.py --dir $MUSE_ROOT/dataset/HDTF/source
  ```
  可选 `--remove_zip` 解压后删除 zip。完成后用该目录作 `prepare_distill_data.py` 的 `--video_dir`。

### 1.2 无法访问 Hugging Face 时的替代来源

服务器无法直连 Hugging Face 时，可用以下方式之一准备视频：

| 方式 | 说明 |
|------|------|
| **本机/代理下载后上传** | 在能访问 Hugging Face 的机器（本机或代理）用 `huggingface-cli download` 或网页下载 HDTF，再通过 scp/rsync 传到服务器。 |
| **VoxCeleb2** | 说话人视频数据集，国内有研究者将分卷上传至**百度网盘**（可搜「VoxCeleb2 国内下载」）。下载后按说明合并解压，得到大量 MP4，取其中若干作为 `--video_dir` 即可。需自行遵守数据集使用条款。 |
| **OpenDataLab / ModelScope** | 见下 **1.2.1 ModelScope 推荐**。 |
| **自备视频** | 见下 1.3，任意单人正面说话、带音轨的 MP4 即可，无需固定数据集。 |

### 1.2.1 ModelScope 推荐：数据堂 1998 人唇语视频数据

**数据集**：数据堂—1,998 人唇语视频数据（人脸 + 口型 + 说话视频，适合对话头/口型同步）。

- **页面**：<https://www.modelscope.cn/datasets/DatatangBeijing/1998People-LipLanguageVideoData>
- **下载**（在服务器上执行）：

```bash
# 安装 ModelScope（若未安装）
pip install modelscope

# 进入一个工作目录，用 Python 拉取数据集到本地
python -c "
from modelscope.msdatasets import MsDataset
# 下载到当前目录下的 1998People-LipLanguageVideoData
ds = MsDataset.load('DatatangBeijing/1998People-LipLanguageVideoData', split='train')
# 查看第一条，确认数据结构和路径
print(next(iter(ds)))
# 若需把视频集中到同一目录供 prepare_distill_data.py 使用，可遍历 ds 复制/链接到目标目录
"
```

- 下载完成后，数据集会缓存在 ModelScope 默认目录（如 `~/.cache/modelscope/hub/datasets/...`），或在 `MsDataset.load(..., cache_dir='./my_data')` 指定目录。将得到的 **视频文件集中到同一目录**（例如 `$MUSE_ROOT/dataset/distill_videos/`），再将该目录作为 `--video_dir` 传给 `prepare_distill_data.py`。
- **使用条款**：数据堂数据集通常需遵守非商业/科研用途等条款，使用前请在页面上确认许可说明。

若该数据集需申请或仅部分公开，可在 ModelScope 搜索「唇语」「人脸视频」「说话人」等关键词，选用带国内可下链接的其它数据集，按同样方式下载并整理成同一目录的 MP4 即可。

### 1.3 自备视频

若无 HDTF，可用任意**单人正面说话视频**（带人脸、带音轨），分辨率与时长不限，建议 ≥25fps。将视频放到同一目录，例如 `$MUSE_ROOT/data/videos/`，后续用「二」中的脚本按「每个视频 → 一个 avatar」生成。

---

## 二、生成 Avatar 列表与配置（每个视频一个 avatar）

蒸馏需要多个 avatar（如 avator_1, avator_2, …），每个对应一段视频。用本仓库提供的脚本从**视频列表**生成：

- MuseTalk 用的 `realtime.yaml` 风格配置（每个 avatar 的 `video_path`、`preparation: True`、`audio_clips`）；
- 从每条视频抽出的音频 `data/audio/<id>.wav`；
- `dataset/distill/train_avatars.txt`（avatar_id 一行一个）。

**步骤**（在 **tad 仓库根** 执行）：

```bash
cd $REPO   # tad 仓库根

# 1）生成配置 + 抽音频 + train_avatars.txt
#    --video_dir: 放 MP4 的目录（如 HDTF source 或自备目录）
#    --max_avatars: 最多用多少个视频（默认 20）
python step3/distill/prepare_distill_data.py \
    --video_dir /path/to/MuseTalk/dataset/HDTF/source \
    --muse_root  /path/to/MuseTalk \
    --max_avatars 20
```

脚本会：

- 在 `--video_dir` 下扫描 `.mp4`，按文件名排序后依次编号 1, 2, …；
- 为每个视频提取音频到 `$MUSE_ROOT/data/audio/<id>.wav`；
- 在 `$MUSE_ROOT/` 下生成 `configs/inference/realtime_distill.yaml`（avator_1, avator_2, …，每个 `preparation: True`，`audio_clips` 指向对应 wav）；
- 在 `$MUSE_ROOT/dataset/distill/train_avatars.txt` 写入 `avator_1` ~ `avator_<n>`。

若没有 HDTF，可把自备视频目录传给 `--video_dir`，同样会按「一视频一 avatar」处理。

---

## 三、在 MuseTalk 中跑 Avatar 预处理（生成 latents.pt）

在 **MuseTalk 项目根** 执行（需已安装 MuseTalk 依赖、已下载 VAE/UNet 等模型）：

```bash
cd $MUSE_ROOT

# 使用上一步生成的配置，仅做 preparation（生成 results/v15/avatars/<id>/latents.pt）
python scripts/realtime_inference.py \
    --inference_config configs/inference/realtime_distill.yaml \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --version v15
```

脚本会对配置里每个 avatar 执行 `preparation=True` 的初始化：读视频 → 抽帧、人脸检测、VAE 编码 → 写出 `results/v15/avatars/<avatar_id>/latents.pt` 等。若某条视频报错（无人脸、解码失败等），可删掉该视频或改配置后重跑。

**注意**：`realtime_inference` 会为每个 avatar 再跑一遍推理并写出一条结果视频；若只想做蒸馏数据，可跑完后忽略输出视频，只保留 `results/v15/avatars/` 下的 `latents.pt`。

---

## 四、预计算音频特征（生成 dataset/distill/audio_feats/）

仍在 **MuseTalk 项目根**：

```bash
cd $MUSE_ROOT
PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/precompute_audio_feats.py \
    --avatar_list dataset/distill/train_avatars.txt \
    --out_dir     dataset/distill/audio_feats/ \
    --audio_dir   data/audio/
```

- `precompute_audio_feats.py` 会按 `train_avatars.txt` 中的 `avator_<id>` 找 `data/audio/<id>.wav`（与「二」中抽出的音频一致），生成 `dataset/distill/audio_feats/avator_<id>.pt`。
- 若某条没有对应 wav，会报错或跳过，请确认「二」中已正确抽音频。

---

## 五、检查是否就绪

在 `$MUSE_ROOT` 下检查：

```bash
# 每个 avatar 应有 latents.pt
ls results/v15/avatars/avator_*/latents.pt

# 每个 avatar 应有音频特征
ls dataset/distill/audio_feats/avator_*.pt

# 列表一致
cat dataset/distill/train_avatars.txt
```

若三者数量、命名一致，即可启动蒸馏训练：

```bash
cd $MUSE_ROOT
PYTHONPATH=$MUSE_ROOT accelerate launch --num_processes 4 \
    $REPO/step3/distill/train_distill.py \
    --config    $REPO/step3/distill/configs/distill.yaml \
    --student_config $REPO/step3/distill/configs/student_musetalk.json \
    --avatar_list dataset/distill/train_avatars.txt
```

---

## 六、目录结构小结

```text
$MUSE_ROOT/
├── dataset/
│   ├── HDTF/source/          # （可选）HDTF 原始 MP4
│   └── distill/
│       ├── train_avatars.txt # avatar 列表，由 prepare_distill_data.py 生成
│       └── audio_feats/      # precompute_audio_feats.py 输出
│           ├── avator_1.pt
│           └── ...
├── data/audio/               # 每个视频一条 wav，由 prepare_distill_data.py 抽取
│   ├── 1.wav
│   └── ...
├── results/v15/avatars/      # realtime_inference.py (preparation) 输出
│   ├── avator_1/
│   │   └── latents.pt
│   └── ...
└── configs/inference/
    └── realtime_distill.yaml # prepare_distill_data.py 生成，供 realtime_inference 使用
```

$REPO 为 tad 仓库根（含 `step3/distill/`）。
