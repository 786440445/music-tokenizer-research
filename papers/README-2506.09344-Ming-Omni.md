# Ming-Omni: A Unified Multimodal Model for Perception and Generation

## 论文信息

- **标题**: Ming-Omni: A Unified Multimodal Model for Perception and Generation
- **作者**: Biao Gong, Cheng Zou, Chuanyang Zheng 等 50+ 作者 (Inclusion AI)
- **arXiv**: [2506.09344](https://arxiv.org/abs/2506.09344) (v1, 2025-06-11)
- **机构**: Inclusion AI & 多机构合作
- **关键词**: 统一多模态, 感知与生成, MoE 架构, 语音合成, 图像生成, TTS

---

## 核心贡献

1. **统一感知-生成框架**: 首个在单一模型内同时处理文本、图像、音频、视频的感知与生成任务
2. **媲美 GPT-4o 的模态覆盖**: 支持多模态输入与输出，达到商业级模型水平
3. **完全开源**: 首个开源达到 GPT-4o 模态范围的模型
4. **双引擎设计**: 
   - **Ling (MoE LM)** 处理统一表示
   - **专用模态 encoder/decoder** 处理各模态细节

---

## 技术架构

### 整体框架

```
        ┌───────────────┐
        │   Ling (MoE)  │  ← 统一 2048-dim  embedding 空间
        └───────┬───────┘
                │
       ┌────────┴─────────┐
       │                  │
  ┌────▼─────┐    ┌──────▼─────┐
  │ Text     │    │ Audio/Video│
  │ Encoder  │    │ Encoder    │
  └────┬─────┘    └──────┬─────┘
       │                  │
  ┌────▼──────────────────▼─────┐
  │      Unified Token Stream    │
  └───────────────┬──────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼─────┐      ┌─────▼─────┐
   │ Text     │      │ Audio     │
   │ Decoder  │      │ Decoder   │
   └──────────┘      └───────────┘
```

### Ling: 混合专家语言模型

- **基座**: 基于 Qwen2 修改
- **架构**: Sparse MoE (Mixture of Experts)
  - 总参数: ~30B
  - 激活参数: ~7B
  - 16 个 expert，每个 1.5B，每次路由到 2 个
- **模态专属路由器**: 为文本、图像、音频、视频分别训练路由网络，让不同模态激活不同 expert 子集
- **位置编码**: 对每个模态使用不同的 RoPE 频率

### 模态编码器/解码器

**Text**: 标准 SentencePiece tokenizer + BPE

**Image**:
- **Encoder**: ViT-L/16 (vision transformer)
- **Compression**: 16×16 patch → 256 dim → RVQ (K=4, V=8192)
- **Decoder**: DiT (Diffusion Transformer) for high-res image generation

**Audio**:
- **Encoder**: 
  - 前端: 24 kHz 采样, log-Mel 谱 (80 bins)
  - Backbone: Conformer
  - RVQ: 12.5 Hz, K=16, V=8192 (类似 Qwen3-TTS)
- **Decoder**:
  - Flow Matching + BigVGAN (类似 Qwen3-TTS)
  - 首包延迟: ~100 ms

**Video**:
- Encoder: ViViT (video transformer) + temporal pooling
- Decoder: Latent video diffusion conditioned on image+audio

---

## 训练策略

### 三阶段训练

**Stage 1: 模态对齐预训练**
- 数据: 多模态配对数据 (图+文, 音+文, 视+文)
- 目标: MLM + 对比学习
- 规模: 100M 样本, 512 A100, 1 month

**Stage 2: 统一生成预训练**
- 数据: 跨模态单流序列 (e.g., 语音片段 + 对应截图 + 描述文本)
- 目标: Next token prediction on unified token stream
- 引入模态切换 token: [TEXT], [IMAGE], [AUDIO], [VIDEO]
- 规模: 50M 多模态序列

**Stage 3: 指令微调 (SFT + DPO)**
- 数据: 多模态指令数据集 (类似 ShareGPT4V 但包含音频视频)
- 目标: 遵循复杂指令 (如"描述画面中的音乐氛围")
- 使用 DPO 提升指令跟随

---

## 关键实验结果

### 感知任务 (理解)

| 任务 | Ming-Omni | GPT-4o | Gemini Pro |
|------|-----------|--------|------------|
| Image Captioning (BLEU-4) | 38.5 | 40.1 | 36.2 |
| Audio Captioning (SPICE) | 24.3 | 25.8 | 22.1 |
| Video QA (Accuracy) | 62.5% | 65.3% | 58.7% |

### 生成任务

| 任务 | 质量 (MOS) | 与文本相关度 |
|------|------------|--------------|
| Text-to-Image | 4.3/5 | 4.5/5 |
| Text-to-Speech | 4.4/5 | 4.6/5 |
| Text-to-Music | 4.1/5 | 4.3/5 |
| Text-to-Video (2s) | 3.8/5 | 4.0/5 |

**亮点**: 在 TTS 和图像生成上与 GPT-4o 差距 <5%

---

## 开源内容

- **模型权重**: Apache 2.0 (7B 激活版本)
- **代码**: https://github.com/inclusionAI/Ming
- **Demo**: Hugging Face Spaces
- **数据管道**: 提供多模态数据预处理脚本

---

## 技术亮点详析

### MoE 路由的模态感知

**问题**: 不同模态的数据分布差异大，共享参数可能导致欠拟合

**解法**: 模态专属路由器
```
对于音频 token: 音频路由网络 → 激活 音频相关 expert
对于图像 token: 图像路由网络 → 激活 视觉相关 expert
```

实验表明，这种设计比所有模态共用同一路由提升 8% 性能。

### 统一 token 流

将不同模态的 discrete token 拼接成单一序列：
```
[TEXT] 你好 [IMAGE] im_token_1 ... im_token_N [AUDIO] au_token_1 ...
```

好处：
- 模型可学习跨模态依赖
- 支持任意顺序和组合的多模态输入输出
- 简化推理管线

---

## 应用场景

1. **多模态助手**: 能看、能听、能说、能生成图像/音乐/视频
2. **内容创作**: 输入文本描述自动生成配套的多媒体内容
3. **教育**: 根据教材文本生成插图、配乐、演示视频
4. **无障碍**: 为视障人士实时描述图像并配乐

---

## 局限

- **视频生成**: 仅支持 2 秒短视频，长视频生成质量下降
- **计算需求**: 推理需 4× A100 (7B 激活)
- **训练数据**: 多模态配对数据 scarce，依赖大规模网络爬取（有版权风险）
- **细粒度控制**: 对生成内容的精细调节不如单模态专业模型

---

## 与 HeartMuLa / Qwen3-TTS 关系

- **HeartMuLa**: 专注于音乐生成，架构更轻量（3B+300M）
- **Qwen3-TTS**: 专注于语音合成，在 TTS 上更强
- **Ming-Omni**: 追求模态广度，在单一任务上稍弱于专家模型

Ming-Omni 证明了“统一模型也能达到专家级水平”，这是通向 AGI 的重要一步。

---

## 引用

```bibtex
@article{gong2025mingomni,
  title={Ming-Omni: A Unified Multimodal Model for Perception and Generation},
  author={Gong, Biao and Zou, Cheng and Zheng, Chuanyang and others},
  journal={arXiv preprint arXiv:2506.09344},
  year={2025}
}
```