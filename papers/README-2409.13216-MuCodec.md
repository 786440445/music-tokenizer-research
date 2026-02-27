# MuCodec: Ultra Low-Bitrate Music Codec

## 论文信息

- **标题**: MuCodec: Ultra Low-Bitrate Music Codec
- **作者**: Yaoxun Xu, Hangting Chen, Jianwei Yu 等 (清华大学深圳国际研究生院、腾讯 AI Lab)
- **arXiv**: [2409.13216](https://arxiv.org/abs/2409.13216) (v1, 2024-09-20; v3, 2025-07-11)
- **会议**: 待发表
- **关键词**: 超低比特率, 音乐编解码, 语义编码, Flow Matching

---

## 核心贡献

1. **超低比特率**: 在 **0.35 kbps** 比特率下实现可接受的音乐重建质量（此前工作 >1 kbps）
2. **语义-声学解耦**: 使用 MuEncoder 分别 modeling 语义和声学特征
3. **Flow Matching 重建**: 相比 GAN 方法训练更稳定，避免模式崩溃
4. **歌词识别辅助**: 引入歌词识别损失增强语义保持
5. **可扩展比特率**: 支持 0.35 kbps → 1.33 kbps 连续调节

---

## 技术架构

### 整体流程

```
原始音频 → MuEncoder → RVQ → Flow Matching → Mel-VAE → HiFi-GAN → 重建音频
          ↓             ↓          ↓
    语义+声学    离散码本     连续潜在
```

### MuEncoder（核心创新点）

**目的**: 提取既包含语义又包含声学的丰富表示

**架构**:
- 13 层 Conformer blocks
- 结合卷积的局部敏感性 + Transformer 的长程依赖

**两阶段训练**:

**Stage 1: Masked Language Model (MLM) 预训练**
- 随机掩蔽 15-30% 帧
- 预测被掩蔽区域的 token
- 作用: 学习上下文表示

**Stage 2: 多任务联合训练**
- **重建损失** (权重 1.0): 恢复 Mel 频谱和 CQT 特征
- **歌词识别损失** (权重 0.2):
  - CTC loss + RNN-T loss
  - 确保 latent 包含可识别的语音内容
  - **关键点**: 即使纯音乐也加入歌词损失，增强语义

**特征层级选择** (消融实验结果):
| MuEncoder 层 | ViSQOL | SPK_SIM | WER |
|--------------|--------|---------|-----|
| 第 3 层 (低层) | 3.8 | 0.89 | 14.2% |
| 第 7 层 (**中高层**) | **3.6** | **0.85** | **12.5%** |
| 第 11 层 (高层) | 3.2 | 0.78 | 10.8% |

**结论**: 第 7 层作为最佳平衡（第3层背景好但人声差，第11层人声好但背景弱）

### RVQ 量化

**配置**:
- **超低比特率 (0.35 kbps)**: K=1, V=16384
- **高比特率 (1.33 kbps)**: K=4, V=10000

**与 EnCodec 对比**:
- EnCodec: K=32, V=1024 → 总容量差异大
- MuCodec: 更大单码本 + 更少码本层 → 适合超低比特率

### Flow Matching 重建

**为什么不用 GAN?**
- GAN 训练不稳定，容易模式崩溃
- 在超低比特率下更难训练

**Flow Matching 优势**:
- 概率建模，训练稳定
- 所需训练步数少
- 可结合 Reflow 蒸馏加速推理

**架构**:
- Diffusion Transformer (DiT): 24 层 Transformer 2D
- 注意力头维度: 72
- 预测目标: Mel-VAE 的 latent (非直接预测 Mel 谱)
- 推理步数: 50 (蒸馏后可降至 10)

**Mel-VAE + HiFi-GAN**:
- 预训练的连续音频编解码器
- 将 Flow Matching 输出的 latent 解码为 waveform

---

## 训练配置

- **数据集**: 大规模内部音乐数据集 (中英文歌曲)
- **采样率**: 32 kHz 最低
- **片段长度**: 35.84 秒
- **测试集**: 500 首歌曲 (250 中文 + 250 英文)

### MuEncoder 训练

- **GPU**: 8 × 40G A100
- **Batch**: 4
- **总步数**: 20k steps (对比实验做 120k 验证稳定性)
- **优化器**: AdamW
- **Loss 权重**:重建 1.0, 歌词识别 0.2

### Flow Matching 训练

- **GPU**: 8 × A100
- **Batch**: 4
- **步数**: 20k steps
- **lr**: 1e-4

---

## 实验结果

### 客观指标对比

| 模型 | 比特率 | ViSQOL ↑ | SPK_SIM ↑ | WER ↓ |
|------|--------|----------|-----------|-------|
| DAC+GAN | 0.35 kbps | 2.8 | 0.62 | 25.4% |
| SemantiCodec | 0.35 kbps | 3.0 | 0.71 | 18.2% |
| **MuCodec** | **0.35 kbps** | **3.6** | **0.85** | **12.5%** |
| DAC+GAN | 1.33 kbps | 3.4 | 0.78 | 15.2% |
| SemantiCodec | 1.33 kbps | 3.5 | 0.82 | 14.1% |
| **MuCodec** | **1.33 kbps** | **4.0** | **0.91** | **8.5%** |

**结论**: 在相同比特率下，MuCodec 全面优于基线

### 主观评估 (MUSHRA)

- **0.35 kbps**: MuCodec 82.7 vs SemantiCodec 68.5 vs DAC+GAN 52.3
- **1.33 kbps**: MuCodec 87.4 vs SemantiCodec 76.2 vs DAC+GAN 65.8

### 消融研究

**歌词识别损失的重要性**:

| 配置 | ViSQOL | SPK_SIM | WER |
|------|--------|---------|-----|
| 仅 MLM | 3.1 | 0.76 | 18.3% |
| +重建损失 | 3.4 | 0.81 | 15.6% |
| **+歌词损失** | **3.6** | **0.85** | **12.5%** |

**发现**: 歌词损失显著降低 WER，证明其增强语义保持

---

## 技术影响

1. **超低比特率可行性**: 首次证明 0.35 kbps 下仍可保持可接受质量
2. **语义引导重建**: 歌词识别作为辅助任务的有效性
3. **流式潜力**: 虽然本文未强调流式，但架构可适配因果设计（HeartCodec 继承此思路）
4. **HeartMuLa 基础**: MuEncoder 成为 HeartMuLa 的重要组成部分

---

## 局限

- **未完全开源**: 论文称开源代码但仓库未公开（截至本文撰写）
- **比特率范围**: 仅覆盖 0.35-1.33 kbps，更高比特率未探索
- **流式支持**: 未明确说明是否支持实时流式推理
- **多语言**: 主要针对中英，其他语言未测试

---

## 与 HeartCodec 关系

MuCodec 是 HeartMuLa 团队的前期工作，HeartCodec 继承并改进:

| 方面 | MuCodec | HeartCodec |
|------|---------|------------|
| 帧率 | 25 Hz | **12.5 Hz** (更低) |
| RVQ 层数 | 1-4 层 | **8 层** (更高容量) |
| 语义编码器 | MuEncoder | MuEncoder + WavLM + Whisper |
| 重建方法 | Flow-Matching | Flow-Matching + SQ-Codec |
| 流式 | 未强调 | **完全因果** |
| 训练数据 | 内部数据 | **60 万首歌曲** |

---

## 开源状态

- **代码**: 声称开源但 GitHub 链接未提供完整 (截至 2025-07)
- **Demo**: https://xuyaoxun.github.io/MuCodec_demo/
- **模型权重**: 未公开下载

---

## 引用

```bibtex
@article{xu2024mucodec,
  title={MuCodec: Ultra Low-Bitrate Music Codec},
  author={Xu, Yaoxun and Chen, Hangting and Yu, Jianwei and others},
  journal={arXiv preprint arXiv:2409.13216},
  year={2024}
}
```