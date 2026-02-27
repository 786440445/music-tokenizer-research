# StemGen: A Music Generation Model That Listens

## 论文信息

- **标题**: StemGen: A music generation model that listens
- **作者**: Julian Parker, Etienne Thobe, Christian J. Steinmetz (BBC R&D)
- **会议**: ICASSP 2024
- **arXiv**: [2312.08723](https://arxiv.org/abs/2312.08723)
- **机构**: BBC Research & Development
- **关键词**: 音乐生成, 上下文感知, 非自回归, Stem 分离

---

## 核心贡献

1. **上下文感知生成**: StemGen 能够监听已存在的音乐片段，并生成与之协调的新声部（stem）
2. **多轨道条件**: 支持将已有的 drums、bass、guitar 等 tracks 作为条件输入
3. **非自回归架构**: 使用并行生成，避免自回归累积误差， faster sampling
4. **可解释的 control**: 通过已有 tracks 的风格控制生成内容

---

## 技术架构

### 问题定义

不是从头生成完整音乐，而是：
- **输入**: 已有的 K 个 stem (如 drum track, bass track)
- **输出**: 新 stem (如 guitar, synth) 与之协调
- **条件**: 文本描述 + 已有 stem 的 token 序列

### 整体架构

```
已有 stems → Tokenizer (EnCodec) → Cross-Attention LM → 新 stem tokens → 解码
                |                                          |
              (多轨道)                                 (音频生成)
```

### Tokenizer

- **使用**: EnCodec (50 Hz, 12 kbps)
- **多轨道**: 每个 stem 独立编码为 token 序列
- **拼接**: 多 stem 序列按轨道顺序拼接，带轨道 ID embedding

### 生成模型

- **类型**: Transformer decoder (non-autoregressive per-track)
- **条件**: 
  - 文本 embedding (T5)
  - 已有 stem 的 token embedding (cross-attention)
  - 位置编码 + 轨道 embedding
- **并行生成**: 所有时间步同时采样（借助 classifier-free guidance）

### 训练策略

- **目标**: 最大化新 stem token 的条件概率 p(t_new | t_old, text)
- **损失**: 交叉熵 + 对抗损失（optional）
- **teacher forcing**: 训练时使用真实新 stem token

---

## 关键创新点

### 1. 交叉注意力机制

```
Query: 新 stem token positions
Key/Value: 所有已有 stem token + 文本
```

这允许新 stem "关注" 已有 tracks 的任意时刻，实现灵活的协调。

### 2. 轨道感知嵌入

为每个轨道类型（drums, bass, etc.）学习独立的 embedding，帮助模型理解轨道间的角色差异。

### 3. 非自回归采样

使用 **Mask-Predict** 算法:
1. 随机初始化新 stem token mask
2. 并行预测所有位置
3. 根据置信度逐步替换 mask 位置
4. 重复 N 步（如 10 步）

比自回归快 5-10 倍，且避免误差累积。

---

## 训练数据

- **来源**: Multi-track 音乐数据集 (如 MUSDB18, Slakh2100)
- **规模**: ~10,000 首歌曲，每首 4-8 个 stem
- **预处理**:
  - 分离每个 stem
  - EnCodec 编码每个 stem 为 50 Hz token 序列
  - 随机选择 1-3 个已有 stem 作为条件，其余作为生成目标

---

## 实验结果

### 客观指标

| 模型 | FAD ↓ | KL ↓ | 协调度 ↑ |
|------|-------|------|----------|
| MusicGen (单条件) | 11.5 | 1.8 | 0.65 |
| **StemGen** | **9.2** | **1.4** | **0.82** |

**协调度**: 计算生成 stem 与已有 stem 的节奏/和声相似度

### 主观评估 (BBC 内部听审)

- **协调性**: 4.1/5
- **音乐性**: 3.9/5
- **新颖性**: 4.0/5

**优势**: 在已有 tracks 的基础上，能生成自然协调的补充声部

---

## 应用场景

1. **音乐创作辅助**: 给定鼓点，AI 生成贝斯和吉他
2. **混音工具**: 自动补全缺失轨道
3. **游戏音频**: 动态生成音乐层次，随玩家操作实时添加新乐器
4. **remix 制作**: 基于原曲生成新的 remix 版本

---

## 局限

- **需要多轨道数据**: 训练数据 scarce， MUSDB18 仅 150 首全分离歌曲
- **轨道数量限制**: 最多 8 个轨道，超出性能下降
- **风格适应**: 对极端风格（如 free jazz）协调能力弱
- **计算成本**: 交叉注意力在长序列上显存占用高

---

## 开源

- **代码**: https://github.com/bbc/StemGen (未完全开源，部分模块)
- **模型**: 未公开发布权重
- **Demo**: BBC R&D 内部演示

---

## 与 MusicGen/HeartMuLa 对比

| 模型 | 条件方式 | 生成目标 | 架构 | 数据需求 |
|------|----------|----------|------|----------|
| MusicGen | 文本 | 完整音乐 | 自回归 | 单轨道大量数据 |
| StemGen | 已有 stems + 文本 | 单个新 stem | 非自回归 | 多轨道数据 |
| HeartMuLa | 歌词+标签+参考 | 完整歌曲 | 层次 LM | 超大规模 |

**StemGen** 是"增量生成"思路，适合音乐制作工作流

---

*ICASSP 2024 最佳论文候选（ rumor ）*