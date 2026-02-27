# AudioGen: Textually Guided Audio Generation

## 论文信息

- **标题**: AudioGen: Textually Guided Audio Generation
- **作者**: Felix Kreuk et al. (Meta AI)
- **会议**: ICLR 2023 (spotlight)
- **arXiv**: [2209.15352](https://arxiv.org/abs/2209.15352)
- **代码**: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **关键词**: 文本到音频, 自回归, EnCodec, 数据增强

---

## 核心贡献

1. **通用音频生成**: 首个大规模文本到音频（非音乐）生成模型，支持自然声音、环境音、音效
2. **数据增强策略**: 提出 **Temporal Alignment** 和 **Frequency Alignment**，从单样本生成多样训练数据
3. **与 MusicGen 共享架构**: 使用相同的 Transformer LM + EnCodec tokenizer，证明架构通用性
4. **零样本迁移**: 在未见过的声音类别上仍有较好表现

---

## 技术架构

### 整体流程

```
文本 → T5-XXL 编码 → Transformer LM (1B params) → EnCodec tokens → 解码 → 波形
```

### 关键设计

- **Tokenizer**: EnCodec (50 Hz, 12 kbps)
- **文本编码**: T5-XXL 冻结特征，仅训练少量适配层
- **条件机制**: 文本 token 以前缀形式注入
- **自回归解码**: 逐 token 生成音频 sequence

### 数据增强（重点创新）

**问题**: 真实世界的声音标注少，每个类别样本有限

**方案**:
1. **Temporal Alignment**: 同一事件在不同时间点标注 → 通过时间偏移生成新样本
2. **Frequency Alignment**: 同一声音在不同频谱分布 → 通过频域变换生成新样本

**效果**: 将每个原始样本拓展为 10-20 个增强样本，显著提升少样本学习

---

## 训练细节

- **数据集**: AudioSet (2M clips) + 内部数据 (~10M)
- **时长**: 最多 10 秒 clips
- **模型规模**: 1B 参数
- **训练资源**: 128 A100 GPUs, 1 week
- **优化**: AdamW, lr=1e-4, gradient clipping

---

## 实验结果

### 客观指标 (FAD ↓)

| 模型 | AudioCaps | AudioSet | 平均 |
|------|-----------|----------|------|
| AudioGen | 9.8 | 8.5 | 9.15 |
| w/ Data Aug | **7.2** | **5.8** | **6.5** |
| baselines | 15-25 | 12-20 | - |

### 主观评估

- **相关性** (文本-音频): 3.8/5
- **质量** (音频保真): 4.0/5
- **多样性**: 4.1/5

**关键发现**: 数据增强在低资源类别提升最显著 (+30% FAD improvement)

---

## 技术影响

1. **证明了通用性**: 同一架构可同时用于音乐 (MusicGen) 和环境声音 (AudioGen)
2. **数据增强范式**: 为少样本音频生成提供了有效策略
3. **推动 AudioCraft**: 成为 Meta AudioCraft 套件的核心组件

---

## 局限

- **生成长度限制**: 最长 10 秒，长音频需重复拼接（可能不连贯）
- **语义理解有限**: 对复杂文本描述（如“狗在远处叫”）理解不够准确
- **计算需求高**: 自回归生成慢，实时应用受限

---

## 开源

- **模型**: `facebook/audiogen` (Hugging Face)
- **库**: `audiocraft` (pip installable)
- **Demo**: Hugging Face Spaces

---

## 与 MusicGen 关系

- **共享 Tokenizer**: EnCodec
- **共享 LM 架构**: Decoder-only Transformer
- **差异**: MusicGen 额外支持旋律条件和结构标签

---

*Published: ICLR 2023 (oral/spotlight)*