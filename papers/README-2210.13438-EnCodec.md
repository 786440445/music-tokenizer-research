# EnCodec: High Fidelity Neural Audio Compression

## 论文信息

- **标题**: High Fidelity Neural Audio Compression
- **作者**: Alexandre Défossez et al. (Meta AI)
- **arXiv**: [2210.13438](https://arxiv.org/abs/2210.13438)
- **会议**: NeurIPS 2022 (oral)
- **代码**: [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **关键词**: 神经音频编解码, RVQ, 实时, 流式

---

## 核心贡献

1. **首个实时高保真神经音频编解码器**：在 24 kHz 采样率下，12 kbps 比特率达到透明质量（主观 MOS 4.5/5）
2. **流式支持**：设计完全因果的编解码器，支持低延迟实时传输（<50 ms）
3. **残差向量量化 (RVQ)**：多层 RVQ 结构，在保持极低比特率的同时最大化重建质量
4. **多尺度频谱损失**：结合 STFT、Mel 频谱、对抗损失，确保全频带保真

---

## 技术架构

### 整体 pipeline

```
原始波形 → Encoder → RVQ → Decoder → 重建波形
          (CNN+RNN)   (K=32)   (CNN+CNN)
```

### 编码器

- **输入**: 24 kHz 单声道/立体声
- **结构**: 
  - 1D-CNN 下采样 (stride 2×5层) → 192 维向量
  - 双向 LSTM (2层) → 上下文感知
  - 投影层 → 128 维向量 z
- **帧率**: 50 Hz (每 20 ms 一帧)

### 残差向量量化 (RVQ)

**核心创新**: 
- K=32 层 RVQ，每层码本大小 V=1024
- 总容量 ≈ 1024³²（天文数字）
- 第 i 层量化第 i-1 层的残差:
  ```
  r₁ = z - q₀(z)
  q₁(z) = q₀(z) + q₁(r₁)
  r₂ = r₁ - q₁(r₁)
  q₂(z) = q₁(z) + q₂(r₂)
  ...
  ```
- 每层独立训练，使用 straight-through estimator

### 解码器

- **输入**: 量化后的 RVQ 嵌入
- **结构**: 
  - 转置 CNN 上采样 (stride 2×5层)
  - 多尺度谱图判别器 (Multi-Scale STFT Discriminator)
- **输出**: 24 kHz 波形

### 损失函数

1. **重建损失**:
   - Waveform L1 loss
   - Mel 频谱 L1 loss
   - STFT 多分辨率 loss

2. **对抗损失**:
   - Multi-Scale STFT Discriminator
   - Feature Matching loss

3. **RVQ 承诺损失**:
   - L_commit = ||z - q(z)||²

4. **速率控制**:
   - 可丢弃高层码本实现可变比特率 (3-12 kbps)

---

## 性能对比

### 比特率 vs 质量

| 比特率 | STOI ↑ | PESQ ↑ | ViSQOL ↑ | 主观 MOS |
|--------|--------|--------|----------|----------|
| 3 kbps | 0.92   | 2.8    | 3.5      | 3.8/5    |
| 6 kbps | 0.95   | 3.4    | 4.1      | 4.2/5    |
| 12 kbps| **0.98** | **4.2** | **4.7**  | **4.5/5** |

**关键结果**:
- 12 kbps 达到透明质量（MOS 4.5，接近原始音频 4.7）
- 比特率仅为传统 MP3 (128 kbps) 的 1/10，质量相当

### 实时性能

- **编码延迟**: 6.5 ms (A100 GPU)
- **解码延迟**: 4.2 ms
- **端到端延迟**: <11 ms
- **支持流式**: 逐帧处理，无全局依赖

---

## 技术影响

1. **成为行业标准**: EnCodec 被广泛采用为音频 tokenizer（MusicGen, AudioGen, Voicebox 等）
2. **推动神经编解码**: 证明神经编解码可达到商业应用质量
3. **开源贡献**: 提供高质量实现，促进可重复研究

---

## 后续改进方向（由后续论文发扬）

- **语义增强**: MuCodec 添加 HuBERT 语义特征
- **更低帧率**: HeartCodec 降至 12.5 Hz
- **音乐优化**: HeartMuLa 针对音乐特性微调

---

## 开源

- **PyTorch 实现**: `facebookresearch/encodec`
- **模型权重**: 多种比特率配置
- **Hugging Face**: `facebook/encodec`

---

## 引用

```bibtex
@article{defossez2022high,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={NeurIPS},
  year={2022}
}
```