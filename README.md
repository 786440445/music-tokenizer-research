# Music Tokenizer Research

## 项目概述

本项目收集和整理了截至2026年1月31日的主流开源音乐合成tokenizer相关论文和项目资源。涵盖Meta、Google、OpenAI等主要研究机构的重要工作。同时提供详细的技术方法综述和训练指南。

## 📚 文档资源

### 综述文档
- **[SURVEY.md](SURVEY.md)** - HeartMuLa 与 MuCodec 论文深度分析
- **[TRAINING_METHODS.md](TRAINING_METHODS.md)** - 完整的 Music Tokenizer 训练方法指南

### 主要研究方向

### 音频Tokenizers (音频分词器)

音频tokenizer是将连续音频信号转换为离散token序列的关键组件，是现代音乐生成系统的基础：

- **EnCodec** (Meta) - 高保真神经音频压缩
- **SoundStream** (Google) - 端到端神经音频编解码器

### 音乐生成模型

- **MusicGen** (Meta) - 可控音乐生成的单阶段Transformer模型
- **AudioGen** (Meta) - 文本条件音频生成模型
- **AudioLM** (Google) - 延续性音频生成
- **MusicLDM** - 基于扩散的音乐生成

## 已收集论文

### 1. MusicGen - Simple and Controllable Music Generation
- **作者**: Meta AI (Yossi Adi et al.)
- **arXiv**: [2306.05284](https://arxiv.org/abs/2306.05284)
- **会议**: NeurIPS 2023
- **代码**: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- **核心贡献**: 单阶段Transformer LM，多流压缩音乐表示，高效token交织模式

### 2. AudioGen - Textually Guided Audio Generation  
- **作者**: Meta AI (Felix Kreuk et al.)
- **arXiv**: [2209.15352](https://arxiv.org/abs/2209.15352)
- **会议**: ICLR 2023
- **核心贡献**: 文本条件音频生成，数据增强技术，多流建模

### 3. EnCodec - High Fidelity Neural Audio Compression
- **作者**: Meta AI (Alexandre Defossez et al.)
- **arXiv**: [2210.13438](https://arxiv.org/abs/2210.13438)
- **核心贡献**: 实时高保真音频编解码器，残差向量量化器，多尺度频谱判别器

### 4. SoundStream - An End-to-End Neural Audio Codec
- **作者**: Google (Neil Zeghidour et al.)
- **arXiv**: [2107.03312](https://arxiv.org/abs/2107.03312)
- **核心贡献**: 3kbps-18kbps可变比特率，结构化dropout，实时智能手机推理

### 5. MusicLDM - Enhancing Novelty in Text-to-Music Generation
- **作者**: Ke Chen et al.
- **arXiv**: [2308.01546](https://arxiv.org/abs/2308.01546)
- **核心贡献**: 节拍同步Mixup策略，数据增强，避免版权问题

### 6. StemGen - A music generation model that listens
- **作者**: Julian Parker et al.
- **arXiv**: [2312.08723](https://arxiv.org/abs/2312.08723)
- **会议**: ICASSP 2024
- **核心贡献**: 音乐上下文响应生成，非自回归Transformer架构

### 7. Ming-Omni - A Unified Multimodal Model for Perception and Generation
- **作者**: Inclusion AI (Biao Gong 等 50+ 作者)
- **arXiv**: [2506.09344](https://arxiv.org/abs/2506.09344)
- **核心贡献**: 统一多模态模型，支持文本、图像、音频、视频的感知与生成；具备语音合成和图像生成能力；是首个达到 GPT-4o 模态覆盖范围的开源模型
- **本地文件**: [papers/ming_omni_2506.09344.pdf](./papers/ming_omni_2506.09344.pdf)
- **代码**: https://github.com/inclusionAI/Ming

### 8. Qwen3-TTS - Qwen3-TTS Technical Report
- **作者**: 阿里云 Qwen 团队 (Hangrui Hu 等)
- **arXiv**: [2601.15621](https://arxiv.org/abs/2601.15621)
- **核心贡献**: 先进的多语言、可控、鲁棒、流式文本到语音模型；支持3秒语音克隆和描述控制；双 tokenizer 策略（25Hz 单码本 + 12Hz RVQ）；500万小时10语种训练；首包延迟 97ms
- **本地文件**: [papers/qwen3_tts_2601.15621.pdf](./papers/qwen3_tts_2601.15621.pdf)
- **开源协议**: Apache 2.0
- **代码**: https://github.com/QwenLM/Qwen3-TTS

### 9. HeartMuLa - A Family of Open Sourced Music Foundation Models
- **作者**: Dongchao Yang 等 25 位作者
- **arXiv**: [2601.10547](https://arxiv.org/abs/2601.10547)
- **核心贡献**: 开源音乐基础模型家族；包含 HeartCodec (12.5 Hz 低帧率高保真编解码器)、HeartCLAP (音频-文本对齐)、HeartTranscriptor (歌词识别)、HeartMuLa (LLM歌曲生成)；6分钟长音乐生成；细粒度风格控制
- **本地文件**: [papers/heartmula_2601.10547.pdf](./papers/heartmula_2601.10547.pdf) (已存在)
- **代码**: 待开源

### 10. Semantic-Codec - A Low-bitrate and Semantic-rich Audio Codec Tokenizer
- **作者**: Dongchao Yang, Songxiang Liu, Haohan Guo 等
- **arXiv**: [2504.10344](https://arxiv.org/abs/2504.10344)
- **核心贡献**: 用于音频语言建模的低比特率、语义丰富的音频编解码器 tokenizer；声学与语义特征解耦；Flow-Matching 重建策略；在超低比特率下保持高质量
- **本地文件**: [papers/music_tokenizer_2504.10344.pdf](./papers/music_tokenizer_2504.10344.pdf)
- **相关**: HeartMuLa 团队前期工作

## 相关开源项目

### Meta AudioCraft
- **GitHub**: https://github.com/facebookresearch/audiocraft
- **包含**: MusicGen, AudioGen, EnCodec等模型
- **功能**: 音乐生成、音频压缩、文本转音频

### Google AudioLM
- **项目页面**: https://google-research.github.io/seanet/audiolm/
- **功能**: 音频续生、语音生成

## 技术架构

```
原始音频 → Tokenizer (EnCodec/SoundStream) → 离散Token序列 
→ Transformer LM (MusicGen/AudioLM) → 生成Token序列 
→ Tokenizer解码 → 合成音频
```

## 📖 技术综述：音乐 Tokenizer 演进路线

### 核心挑战

音乐 tokenization 面临三重矛盾：
1. **低帧率 vs. 细节保留**：帧率越低，序列越短，但可能丢失瞬态细节
2. **低比特率 vs. 重建质量**：压缩率高，但音质下降
3. **语义丰富 vs. 声学保真**：语义信息vs.频谱细节，难以兼得

### 技术演进三阶段

#### Stage 1: 基础 RVQ 时代（2022-2023）

**代表**: EnCodec, SoundStream

- 单层或简单残差向量量化
- 帧率：25-50 Hz
- 比特率：3-18 kbps
- 优势：实时、稳定
- 劣势：容量有限，长程建模弱

#### Stage 2: 语义增强时代（2024-2025）

**代表**: MuCodec, SemantiCodec

- 引入语义编码器（HuBERT/MERT）
- 语义与声学特征解耦
- 比特率降至 0.35-1.33 kbps
- 超低比特率下保持音质

#### Stage 3: 多模态统一时代（2025-2026）

**代表**: Ming-Omni, Qwen3-TTS, HeartMuLa

- 统一框架：tokenizer + LLM + decoder
- 双 tokenizer 策略：
  - 12.5 Hz RVQ（快速流式）
  - 25 Hz 单码本（高质量）
- 500万小时+
- 多语言支持（5-10语种）
- 生成质量达到商业级（Suno 级别）

### 关键创新对比

| 维度 | HeartMuLa | Qwen3-TTS | MuCodec | Ming-Omni |
|------|-----------|-----------|---------|-----------|
| **帧率** | 12.5 Hz | 12/25 Hz | 25 Hz | 多模态(含TTS) |
| **量化** | 8×8192 RVQ | 16×RVQ / 单VQ | 1-4×10k RVQ | Multi-codebook |
| **语义** | Whisper+WavLM | Qwen-Audio | MuEncoder | 多模态融合 |
| **重建** | Flow+SQ-Codec | Block-DiT | Flow-Matching | VAE+GAN |
| **训练数据** | 100k+小时 | 500万小时 | 大规模内部 | 多模态数据 |
| **最大长度** | 6分钟 | 长语音 | 35秒 | 多媒体 |
| **许可证** | Apache 2.0 | Apache 2.0 | 未明确 | 开源 |

### 未来趋势

1. **更低帧率**：挑战 5-10 Hz 下的质量保持
2. **更大容量**：RVQ 层数增加（16→32）
3. **端到端统一**：Tokenizer + LM + Decoder 联合优化
4. **多模态扩展**：音频-文本-图像-视频四模态
5. **实时流式**：<100ms 延迟成为标配

---

## 使用说明

1. 克隆本仓库
2. 进入`papers/`目录阅读相关论文
3. 参考各论文的官方实现进行实验

## 目录结构

```
music-tokenizer-research/
├── README.md
├── SURVEY.md                    # 论文综述
├── TRAINING_METHODS.md          # 训练方法指南
├── papers/
│   ├── musicgen.pdf
│   ├── audiogen.pdf
│   ├── encodec.pdf
│   ├── soundstream.pdf
│   ├── musicldm.pdf
│   ├── stemgen.pdf
│   ├── heartmula_2601.10547.pdf  # HeartMuLa 论文
│   └── mucodec_2409.13216.pdf    # MuCodec 论文
└── references/
```

## 贡献指南

欢迎提交Pull Request补充新的论文和资源！

## 许可证

本项目收集的论文版权归原作者及其出版商所有。
