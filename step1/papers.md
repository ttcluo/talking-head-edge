# 论文阅读清单（第一步必读）

> 阅读顺序：按编号顺序，先必读，后选读。
> 阅读重点：标注了每篇的核心提问，带着问题读，效率高 3 倍。

---

## 第一周必读（共 2 篇，约 4 小时）

### P1. MuseTalk 技术报告 ⭐⭐⭐⭐⭐

**信息**
- 标题：MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling
- 链接：https://arxiv.org/abs/2410.10122
- 作者：腾讯音乐 Lyra Lab（2025）

**阅读重点（带着这 5 个问题读）**
1. UNet 的输入是什么？32 通道怎么来的？（masked latent + mask 各 16 通道）
2. 音频特征如何注入 UNet？（Cross-Attention，维度 384 是 Whisper-tiny 的特征维度）
3. Stage1 和 Stage2 的 loss 分别是什么？各自解决什么问题？
4. 「时空采样」（Spatio-Temporal Sampling）具体指什么？
5. 模型的主要失败案例是什么？（Jitter / 身份保持问题）

**阅读笔记模板**
```
架构核心：
训练数据：
Loss 设计：
主要局限：
与我的研究的关联：
```

---

### P2. EchoMimic 论文 ⭐⭐⭐⭐⭐

**信息**
- 标题：EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditions
- 链接：https://arxiv.org/abs/2407.08136
- 作者：蚂蚁集团（AAAI 2025）
- 代码：https://github.com/antgroup/echomimic

**阅读重点**
1. 它是真正的扩散模型（AnimateDiff backbone），推理需要几步？
2. 音频 + 关键点双条件是怎么设计的？
3. 相比 MuseTalk，它在哪些指标上更好？差距有多大？
4. 推理速度是多少？比 MuseTalk 慢多少倍？
5. 端侧部署的障碍在哪里？

**核心数据记录（读完填入）**
```
推理步数：___
单帧耗时：___ms（在什么 GPU 上）
显存占用：___GB
LSE-C 分数：___
SSIM：___
```

---

## 第二周必读（共 2 篇，约 4 小时）

### P3. ViDiT-Q（视频扩散模型量化）⭐⭐⭐⭐

**信息**
- 标题：ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation
- 链接：https://arxiv.org/abs/2406.02540
- 代码：https://github.com/thu-nics/ViDiT-Q（清华，可直接用）

**阅读重点**
1. W8A8 和 W4A8 量化的精度损失分别是多少？
2. 哪些层对量化最敏感？（找到 Figure/Table 里的层敏感度分析）
3. 混合精度分配策略是什么？
4. 这套方法能不能直接用在 UNet-based 的 MuseTalk 上？需要什么改动？

---

### P4. LightningCP（特征缓存加速）⭐⭐⭐

**信息**
- 标题：Lightning Fast Caching-based Parallel Denoising Prediction for Accelerating Talking Head Generation
- 链接：https://arxiv.org/abs/2509.00052

**阅读重点**
1. 什么特征被缓存了？缓存多久更新一次？
2. Decoupled Foreground Attention（DFA）怎么实现的？
3. 加速比是多少？质量损失多少？
4. 这个方法和量化能否结合？

---

## 选读（根据自己研究方向选择）

### 方向 A：量化方向

#### P5. MixDQ（混合精度量化）⭐⭐⭐
- 链接：https://arxiv.org/abs/2405.17873
- 关注：少步扩散模型的量化，INT4 对质量的影响

#### P6. QuantCache（量化+缓存联合）⭐⭐⭐
- 链接：https://arxiv.org/abs/2503.06545
- 关注：量化和缓存如何联合优化，6.72x 加速比怎么来的

### 方向 B：蒸馏方向

#### P7. TalkingMachines（2 步扩散蒸馏）⭐⭐⭐⭐
- 链接：https://arxiv.org/abs/2506.03099
- 关注：Distribution Matching Distillation 怎么应用到 Talking Head

#### P8. LiveTalk（在线策略蒸馏）⭐⭐⭐
- 链接：https://arxiv.org/abs/2512.23576
- 关注：on-policy 蒸馏在多模态条件下的处理方式

#### P9. REST（流式异步蒸馏）⭐⭐⭐
- 链接：https://arxiv.org/abs/2512.11229
- 关注：ASD（Asynchronous Streaming Distillation）的具体实现

### 方向 C：架构/缓存方向

#### P10. SoulX-FlashHead（端侧高速 Talking Head）⭐⭐⭐⭐
- 链接：https://arxiv.org/abs/2602.07449
- 关注：Oracle-guided 蒸馏，96 FPS 是怎么达到的

#### P11. Teller（CVPR 2025，自回归流式）⭐⭐⭐
- 链接：CVPR 2025 开放仓库
- 关注：自回归框架 vs 扩散模型的速度/质量权衡

---

## 阅读进度记录

| 论文 | 状态 | 完成日期 | 关键结论 |
|------|------|---------|---------|
| P1 MuseTalk 技术报告 | ⬜ 未读 | — | — |
| P2 EchoMimic | ⬜ 未读 | — | — |
| P3 ViDiT-Q | ⬜ 未读 | — | — |
| P4 LightningCP | ⬜ 未读 | — | — |
| P5 MixDQ | ⬜ 未读 | — | — |
| P6 QuantCache | ⬜ 未读 | — | — |
| P7 TalkingMachines | ⬜ 未读 | — | — |
| P8 LiveTalk | ⬜ 未读 | — | — |
| P9 REST | ⬜ 未读 | — | — |
| P10 SoulX-FlashHead | ⬜ 未读 | — | — |
| P11 Teller | ⬜ 未读 | — | — |

---

## 基线数据表（读完 P1 P2 后填入）

| 方法 | 推理步数 | FPS | 显存(GB) | 模型大小(MB) | LSE-C | SSIM |
|------|---------|-----|---------|------------|-------|------|
| MuseTalk 1.0 | 1 | — | — | — | — | — |
| MuseTalk 1.5 | 1 | 30+ | ~8 | ~500 | — | — |
| EchoMimic | 20 | — | — | — | — | — |
| Hallo | — | — | — | — | — | — |
| **本研究目标** | **2-4** | **>15** | **<4** | **<200** | **≈EchoMimic** | — |
