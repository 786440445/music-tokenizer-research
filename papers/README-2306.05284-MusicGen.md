# MusicGen: Simple and Controllable Music Generation

## 论文信息

- **标题**: MusicGen: Simple and Controllable Music Generation
- **作者**: Yossi Adi et al. (Meta AI)
- **会议**: NeurIPS 2023
- **arXiv**: [2306.05284](https://arxiv.org/abs/2306.05284)
- **代码**: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **关键词**: 音乐生成, Transformer, EnCodec, 文本条件

---

## 核心贡献

1. **单阶段架构**：MusicGen 是首个将文本条件、旋律条件、音乐结构标签统一到单一 Transformer LM 的音乐生成模型
2. **高效Token交织**：采用特殊的 token 交织模式（chord/timbre/rhythm tokens），实现细粒度控制
3. **大规模训练**：在 20,000 小时高质量音乐数据上训练，涵盖多种风格
4. ** melodies 支持**：可通过哼唱或 MIDI 旋律引导音乐生成

---

## 技术架构

### 整体流程

```
文本描述 → 文本编码器 → Transformer LM → EnCodec tokens → 解码器 → 音频
          (T5)          (自回归)        (1.6B params)   (EnCodec)
```

### 编码策略

- **文本条件**: T5-XXL 编码文本描述
- **旋律条件**: 将 MIDI/humming 转换为和弦/节奏 token
- **结构标签**: 使用 [intro], [verse], [chorus] 等 token 标记段落
- **Token 交织**: 文本 token、旋律 token、结构 token 按固定模式交错输入

### 解码器

- 使用 **EnCodec** (24 kHz, 12 kbps) 作为音频 tokenizer
- 生成音乐 tokens 后再用 EnCodec decoder 重建波形

---

## 训练细节

- **数据集**: 20,000 小时授权音乐 + 内部数据集
- **Token 率**: EnCodec  operates at 50 Hz (f0) → 1200 tokens/min
- **模型规模**: 1.6B 参数 Transformer (decoder-only)
- **训练时长**: 约 2 weeks on 64 A100 GPUs
- **优化**: AdamW, lr=1e-4, cosine decay

---

## 实验结果

### 客观指标

| 模型 | FAD ↓ | KL Divergence ↓ | 
|------|-------|-----------------|
| MusicGen ( ours ) | 11.5 | 1.8 |
| MusicLM ( Google ) | 13.2 | 2.3 |
| AudioGen | 15.7 | 2.9 |

### 主观评估 (MOS)

- **音乐性**: 4.1/5
- **文本对齐**: 3.9/5
- **整体质量**: 4.2/5

**优势**: 在旋律匹配、结构控制上显著优于基线

---

## 局限性与未来方向

- **生成长度**: 最大支持约 30 秒，长音乐需要分段拼接
- **多语言**: 主要针对英文文本描述
- **实时性**: 自回归生成速度较慢，需优化推理
- **版权**: 训练数据版权问题未完全解决

---

## 开源实现

- **Hugging Face**: `facebook/musicgen`
- **推理库**: `audiocraft`
- **支持功能**: 文本生成、旋律引导、多轨道生成

---

## 与后续工作关系

MusicGen 为后续工作（如 AudioGen、EnCodec 改进）奠定了基础。其单阶段架构思想影响了 HeartMuLa 等 newer models。

---

*Note: 本概述基于 2023 年 NeurIPS 论文版本。*