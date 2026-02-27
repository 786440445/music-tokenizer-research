# HeartMuLa: A Family of Open Sourced Music Foundation Models

## 论文信息

- **标题**: HeartMuLa: A Family of Open Sourced Music Foundation Models
- **作者**: Dongchao Yang, Songxiang Liu, Haohan Guo 等 25 位作者
- **arXiv**: [2601.10547](https://arxiv.org/abs/2601.10547) (v2, 2026-01-26)
- **机构**: 多机构合作（主要来自中国）
- **许可证**: Apache 2.0 (完全开源)
- **代码**: 待开源 (计划中)
- **关键词**: 音乐基础模型, 音乐编解码器, 歌曲生成, 音频-文本对齐

---

## 核心贡献

1. **HeartCodec**: 12.5 Hz 超低帧率、高保真音乐编解码器，在保持质量的同时将序列长度减半
2. **HeartMuLa**: 基于 LLM 的歌曲生成模型，支持长达 6 分钟的长格式音乐创作
3. **HeartCLAP**: 音频-文本对齐模型，用于音乐检索和标签分类
4. **HeartTranscriptor**: 鲁棒的歌词识别模型，针对真实音乐场景优化
5. **细粒度风格控制**: 支持用自然语言控制不同歌曲段落（intro, verse, chorus）的风格
6. **首次复现 Suno 级系统**: 使用学术规模数据和 GPU 资源达到商业级质量

---

## 技术架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    HeartMuLa 生态系统                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ HeartCLAP   │    │ HeartTrans- │    │ HeartCodec      │  │
│  │ 音频-文本   │    │ criptor     │    │ 低帧率音乐编码  │  │
│  │ 对齐模型    │    │ 歌词识别    │    │ (12.5 Hz)       │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                   │          │
│                                          ┌────────▼────────┐│
│                                          │ HeartMuLa       ││
│                                          │ LLM歌曲生成模型 ││
│                                          └─────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## HeartCodec 详解（核心技术）

### 架构三大组件

#### 1. 语义丰富编码器 (Semantic-Rich Encoder)

**多编码器特征融合**:

| 编码器 | 特征类型 | 提取层 | 用途 |
|--------|----------|--------|------|
| MuEncoder | 音乐语义 | 第11层 | 音色、旋律结构 |
| WavLM | 音素级 | 平均6-9层 | 人声发音 |
| Whisper | 语音嵌入 | 全部 | 内容语义 |
| MuEncoder | 声学细节 | 第2层 | 频谱细节 |

所有特征重采样到 25 Hz，通道拼接后线性投影 → y_h

#### 2. 超低帧率压缩器 (Ultra-Low Frame Rate Compressor)

**Query-based 下采样** (核心创新):

1. 在 y_h 每两帧后插入可学习 query token
2. Transformer 编码器处理
3. 每个 query 输出 ↔ 前两帧的摘要
4. 丢弃非 query 帧 → 帧率 25 Hz → **12.5 Hz** → y_l

**残差向量量化 (RVQ)**:
- K=8 码本，V=8192
- 总容量 = 8192⁸（天文数字）
- 输出离散索引 A ∈ [V]^(T×f_l×K)

**损失函数**:
- Commitment loss: L_commit = ||sg(yl) - ŷl||²
- Feature alignment loss: 对齐 MuEncoder 语义和 WavLM 音素特征

#### 3. 高保真重建解码器

**两阶段重建**:
1. **Flow Matching**: 将离散 ŷl 映射到连续 latent z (使用 DiT)
2. **SQ-Codec 解码**: 预训练的连续编解码器从 z 重建波形

**加速**: Reflow 蒸馏将采样步数从 50 降至 10

---

### HeartCodec 训练三阶段

**Stage 1: 预训练 + 微调**
- 数据: ~60 万首歌, 20.48s 片段
- Loss: L₁ = λ_fm L_fm + λ_commit L_commit + λ_align
- GPU: 88 A100, Batch 160, 15 epochs
- lr: 1e-4

**Stage 2: Reflow 蒸馏**
- 数据: 5 万高质量 29.76s 片段
- 仅训练 Flow Matching 模块
- GPU: 8 A100, 2 epochs, lr: 5e-6

**Stage 3: SQ-Codec 微调**
- 数据: 2 万高质量样本 (AudioBox + SongEval 筛选)
- 仅训练 SQ-Codec 解码器
- GPU: 44 A100, 33 epochs, lr: 2e-6

---

## HeartMuLa 歌曲生成模型

### 层次化架构

**双 Transformer 设计**:

1. **Global Transformer (θ_glo, 3B)**:
   - 处理完整序列上下文
   - 预测第 0 码本（粗粒度语义）

2. **Local Decoder (θ_loc, 300M)**:
   - 针对每个时间步
   - 预测残差码本 1-7（细粒度声学）
   - 条件: Global representation + 已预测的码本

**概率分解**:
```
p(a_l | h_<l) = p(a_{l,0} | h_<l) * Π_{k=1}^{K-1} p(a_{l,k} | h_{l,<k}, θ_glo(h_<l))
```

### 条件机制

支持三种条件:

1. **歌词条件**: 标注结构标记 [intro]/[verse]/[chorus]，Llama-3.2 tokenizer
2. **标签条件**: 8 类音乐属性（流派、情绪、乐器等）
3. **参考音频**: 10 秒片段，MuQ-MuLan 提取风格 embedding

### 四阶段渐进训练

| 阶段 | 数据规模 | GPU | epochs | 条件 | 主要目标 |
|------|----------|-----|--------|------|----------|
| Warmup | 10k 小时 | 8 A100 | 5 | C_muq + C_lyrics | 参数收敛 |
| 预训练 | 100k 小时 | 64 A100 | 5 | C_tag + C_muq + C_lyrics | 长程依赖 |
| SFT | 15k 小时 | 8 A100 | 3 | 全部 | 质量提升 |
| DPO | 偏好对 | 8 A100 | 3 | 全部 | 对齐人类偏好 |

---

## 实验结果

### 客观评估 (English)

| 模型 | AudioBox ↑ | SongEval ↑ | Tag-Sim ↑ | PER ↓ |
|------|------------|------------|-----------|-------|
| Suno-v5 | 7.65 | 7.83 | 0.26 | 0.13 |
| MiniMax-2.0 | 7.73 | 7.98 | 0.26 | 0.13 |
| LeVo | 7.55 | 7.79 | 0.13 | 0.22 |
| **HeartMuLa** | **7.55** | **7.82** | **0.26** | **0.09** |

**关键**: 最低 PER (0.09) → 歌词清晰度最佳

### 主观评估 (MOS 百分制)

| 模型 | 音乐性 | 和声 | 结构 | 保真度 | 创造力 | 记忆性 | 文本对齐 | Overall |
|------|--------|------|------|--------|--------|--------|----------|---------|
| Suno-v4.5 | 78.1 | 75.1 | 78.8 | 79.1 | 71.3 | 73.0 | 77.2 | 76.1 |
| **HeartMuLa** | **69.6** | **71.1** | **73.4** | **73.2** | **66.7** | **65.1** | **70.5** | **69.9** |

**注**: HeartMuLa 仍略低于 Suno，但差距在缩小，且开源免费

### 推理加速

- **优化前**: 398.3 秒 (生成 1 首歌)
- **优化后**: 73.4 秒
- **加速比**: **5.4×** (KV-Cache + FlashAttention + CUDA Graph)

---

## HeartCLAP & HeartTranscriptor

### HeartCLAP

- 基于 CLAP 架构改进
- 音乐专用对比学习
- 支持 10+ 音乐风格分类
- 用于检索和标签预测

### HeartTranscriptor

- 针对含背景音乐的歌词识别（ASR 在此场景性能下降）
- 使用分离 + 识别的两阶段设计
- WER 在 MUSDB18 测试集上达到 12.5%

---

## 数据集

**HeartMuLa 训练数据**:
- 总时长: 100,000+ 小时
- 高质量 SFT 子集: 15,000 小时
- 多语言: 英、中、日、韩、西、法、德、意、俄、阿
- 结构标注: 每首歌标注 [intro], [verse], [chorus], [bridge], [outro]
- 风格标签: 基于 MuQ-MuLan 自动打标 (8 大类)

**HeartBeats-Benchmark**: 团队发布的多语言音乐评估基准

---

## 开源与可访问性

- **许可证**: Apache 2.0 (可商用)
- **计划开源**: 模型权重 + 代码 + 数据预处理工具
- **推理 API**: 计划提供云端服务
- **社区**: 欢迎 PR 和 issue

---

## 与 MuCodec / Qwen3-TTS 对比

| 维度 | HeartMuLa | MuCodec | Qwen3-TTS |
|------|-----------|---------|-----------|
| 主任务 | 音乐生成 | 音乐压缩 | 语音合成 |
| 帧率 | 12.5 Hz | 25 Hz | 12/25 Hz |
| 最大长度 | 6 分钟 | 35 秒 | 长语音 |
| 控制方式 | 歌词 + 标签 + 参考音频 | 无 | 参考音频 + 文本描述 |
| 训练数据 | 100k+ 小时 | 百万级小时 | 500 万小时 |
| 多语言 | 10 种 | 中英 | 10 种 |
| 参数规模 | 3B+300M | ~0.5B | 0.6B/1.7B |

**HeartMuLa 特色**: 最完整的开源音乐生成解决方案

---

## 局限

1. **GPU 需求高**: 完整训练需 64 A100 × 数周
2. **音乐性略低于 Suno**: 主观评估仍有 5-10% 差距
3. **极端风格覆盖不足**: 如 free jazz, 实验电子
4. **长音乐结构一致性**: 6 分钟以上可能出现主题漂移
5. **版权数据比例**: 未完全公开训练数据来源

---

## 未来方向

- **更大规模**: 7B → 20B 参数
- **端到端联合训练**: Tokenizer + LM 联合优化
- **用户交互**: 实时反馈循环，在线调优
- **多模态音乐视频**: 生成配套 MV
- **音乐理论集成**: 融入乐理知识图谱

---

## 引用

```bibtex
@article{yang2025heartmula,
  title={HeartMuLa: A Family of Open Sourced Music Foundation Models},
  author={Yang, Dongchao and Liu, Songxiang and Guo, Haohan and others},
  journal={arXiv preprint arXiv:2601.10547},
  year={2026}
}
```

---

**注**: 本概述基于 v2 版本 (2026-01-26)。项目官网: https://heartmu.github.io (待确认)