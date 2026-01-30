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
