# Music Tokenizer 训练方法综述：基于 HeartMuLa 与 MuCodec 的技术指南

## 目录

1. [引言](#引言)
2. [核心技术架构](#核心技术架构)
3. [MuEncoder 设计](#mucoder-设计)
4. [残差向量量化](#残差向量量化)
5. [Flow-Matching 重建](#flow-matching-重建)
6. [训练策略](#训练策略)
7. [数据准备与处理](#数据准备与处理)
8. [训练流程详解](#训练流程详解)
9. [评估指标](#评估指标)
10. [实战代码示例](#实战代码示例)
11. [常见问题与解决方案](#常见问题与解决方案)
12. [总结与建议](#总结与建议)
13. [参考文献](#参考文献)

---

## 引言

本综述基于两篇前沿音乐 Tokenizer 论文的技术方案：

- **HeartMuLa** (arXiv:2601.10547): Meta 等机构提出的开源音乐基础模型家族，核心是 HeartCodec
- **MuCodec** (arXiv:2409.13216): 清华大学与腾讯 AI Lab 提出的超低比特率音乐编解码器

两篇论文的核心目标是：
1. 将高采样率音频压缩为离散 token 序列
2. 保留音乐的语义信息和声学细节
3. 支持高质量的音乐重建和生成

本综述将系统性地总结如何从零训练一个高质量的音乐 Tokenizer。

---

## 核心技术架构

### 整体框架

```
┌─────────────────────────────────────────────────────────────────┐
│                    Music Tokenizer 训练架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  原始Hz, stereo)                                        │
│        音频 (48k │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                            │
│  │   MuEncoder     │  提取声学和语义特征                         │
│  │   (Conformer)   │  - 声学特征：音色、频谱细节                  │
│  │                 │  - 语义特征：旋律、和声、歌词                 │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │      RVQ        │  残差向量量化                               │
│  │  (多码本压缩)   │  - K 个码本，V 个条目                       │
│  │                 │  - 低比特率表示                             │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │ Flow-Matching   │  潜在空间重建                               │
│  │   (DiT 架构)    │  - 噪声到目标的映射                         │
│  │                 │  - 高质量音频生成                           │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │   解码器        │  波形重建                                   │
│  │ (Mel-VAE+HiFi)  │  - Mel 频谱 → 波形                         │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  重建音频                                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | HeartMuLa (HeartCodec) | MuCodec | 推荐选择 |
|------|----------------------|---------|----------|
| 特征编码器 | Whisper + WavLM + MuEncoder | MuEncoder (Conformer) | MuEncoder 独立设计 |
| 帧率 | 12.5 Hz | 25 Hz | 12.5 Hz (更长序列) |
| 量化方式 | RVQ (8×8192) | RVQ (1-4×10000-16384) | 根据比特率需求 |
| 重建方法 | Flow-Matching + SQ-Codec | Flow-Matching + Mel-VAE | 均可 |
| 比特率 | ~1.3 kbps | 0.35-1.33 kbps | 根据场景选择 |

---

## MuEncoder 设计

### 1. 架构选型

#### 1.1 Conformer 架构

Conformer 结合了 Transformer 的注意力机制和卷积的局部特征提取能力：

```python
# Conformer 核心组件
class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, kernel_size=32):
        super().__init__()
        self.ffn = FeedForward(dim, ff_mult)      # 前馈网络
        self.self_attn = MultiHeadAttention(dim, heads)  # 自注意力
        self.conv = ConvolutionalModule(dim, kernel_size)  # 卷积模块
        self.norm = LayerNorm(dim)
    
    def forward(self, x):
        x = x + 0.5 * self.ffn(x)
        x = x + self.self_attn(x)
        x = x + self.conv(x)
        return self.norm(x)
```

**推荐配置**：
- 层数：13-17 层
- 隐藏维度：512-1024
- 注意力头数：8-16
- 卷积核大小：32

#### 1.2 多层级特征提取

MuEncoder 需要提取不同层次的特征：

```python
class MultiLevelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conformer = ConformerStack(config)
        
        # 不同层捕获不同信息
        self.semantic_layer = config.semantic_layer    # 高层：语义
        self.acoustic_layer = config.acoustic_layer    # 低层：声学
        
    def forward(self, x):
        # 获取所有层的特征
        all_layers = self.conformer(x, output_all_layers=True)
        
        # 语义特征（高层）
        semantic = all_layers[self.semantic_layer]
        
        # 声学特征（低层）
        acoustic = all_layers[self.acoustic_layer]
        
        return semantic, acoustic
```

### 2. 两阶段训练策略

#### 2.1 阶段一：MLM 预训练

**目标**：学习上下文感知和基础表示能力

```python
class MLMTraining:
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
    
    def training_step(self, batch):
        audio_features = batch  # shape: [B, T, D]
        
        # 随机掩码
        mask = self.create_mask(audio_features, self.mask_ratio)
        masked_features = audio_features * (1 - mask)
        
        # 预测被掩码的区域
        predictions = self.model(masked_features)
        
        # MSE 损失
        loss = F.mse_loss(predictions[mask], audio_features[mask])
        
        return loss
```

**配置**：
- 掩码比例：15%
- 训练轮数：10-20 epochs
- 学习率：1×10⁻⁴

#### 2.2 阶段二：多任务联合训练

**目标**：同时优化声学和语义表示

```python
class MultiTaskTraining:
    def __init__(self):
        self.recon_weight = 1.0
        self.asr_weight = 0.2
    
    def training_step(self, batch):
        audio, lyrics = batch
        
        # 提取特征
        features = self.encoder(audio)
        
        # 1. 重建损失
        mel_recon = self.decoder_mel(features)
        recon_loss = F.mse_loss(mel_recon, target_mel)
        
        # 2. CQT 重建损失
        cqt_recon = self.decoder_cqt(features)
        cqt_loss = F.mse_loss(cqt_recon, target_cqt)
        
        # 3. ASR 损失（歌词识别）
        asr_output = self.asr_decoder(features)
        asr_loss = self.ctc_loss(asr_output, lyrics)
        
        # 总损失
        total_loss = (self.recon_weight * (recon_loss + cqt_loss) + 
                      self.asr_weight * asr_loss)
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'asr_loss': asr_loss.item()
        }
```

**损失函数配置**：

| 损失类型 | 权重 | 作用 |
|---------|------|------|
| Mel 频谱重建 | 1.0 | 保持声学质量 |
| CQT 特征重建 | 1.0 | 保留频率结构 |
| CTC 损失 | 0.1-0.2 | 捕捉语义信息 |
| RNN-T 损失 | 0.1-0.2 | 增强序列建模 |

### 3. 层选择策略

MuCodec 的消融实验表明：

| 层数 | 特点 | 适用场景 |
|------|------|----------|
| 低层 (3-5) | 强声学特性，背景重建好 | 背景音乐优先 |
| 中层 (7-9) | 平衡声学和语义 | 通用场景 ✓ |
| 高层 (11-13) | 多语义特征，人声清晰 | 歌词优先 |

**推荐**：第7层作为默认选择

---

## 残差向量量化

### 1. RVQ 原理

残差向量量化通过多个码本逐步逼近目标向量：

```python
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_quantizers, codebook_size):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # 多个码本
        self.quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x):
        # 残差量化
        quantized_out = 0
        losses = 0
        indices = []
        
        residual = x
        for quantizer in self.quantizers:
            # 量化当前残差
            quantized, loss, idx = quantizer(residual)
            
            # 累加量化结果
            quantized_out = quantized_out + quantized
            residual = residual - quantized
            
            # 记录
            losses = losses + loss
            indices.append(idx)
        
        return quantized_out, losses, indices
```

### 2. 比特率配置

根据不同的应用场景选择配置：

```python
class RVQConfig:
    # 低比特率配置 (0.35 kbps)
    LOW_BITRATE = {
        'num_quantizers': 1,
        'codebook_size': 16384,
        'frame_rate': 25,
        'bitrate': '0.35 kbps'
    }
    
    # 中等比特率配置 (0.7 kbps)
    MEDIUM_BITRATE = {
        'num_quantizers': 2,
        'codebook_size': 16384,
        'frame_rate': 25,
        'bitrate': '0.7 kbps'
    }
    
    # 高比特率配置 (1.3 kbps)
    HIGH_BITRATE = {
        'num_quantizers': 8,
        'codebook_size': 8192,
        'frame_rate': 12.5,
        'bitrate': '1.3 kbps'
    }
    
    # 超高比特率配置 (2.6 kbps)
    ULTRA_HIGH_BITRATE = {
        'num_quantizers': 8,
        'codebook_size': 8192,
        'frame_rate': 25,
        'bitrate': '2.6 kbps'
    }
```

### 3. 帧率选择

```
帧率对比：

25 Hz:  每秒 25 帧    [██████████] 传统方法
12.5 Hz: 每秒 12.5 帧 [█████]      HeartCodec (推荐)
         ─────────────────────────
         更少的 token，更长的上下文
```

**帧率计算**：

```python
def calculate_bitrate(frame_rate, num_quantizers, codebook_size, sample_rate=48000):
    """计算比特率"""
    bits_per_frame = num_quantizers * np.log2(codebook_size)
    bitrate = (frame_rate * bits_per_frame * sample_rate) / sample_rate
    return bitrate

# 示例
print(f"12.5 Hz, 8×8192: {calculate_bitrate(12.5, 8, 8192):.2f} kbps")
print(f"25 Hz, 8×8192: {calculate_bitrate(25, 8, 8192):.2f} kbps")
```

---

## Flow-Matching 重建

### 1. Flow-Matching 原理

Flow-Matching 是一种比传统扩散模型更稳定的生成方法：

```python
class FlowMatchingDecoder(nn.Module):
    def __init__(self, dim, num_layers=24):
        super().__init__()
        self.dit = DiffusionTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=8,
            head_dim=72
        )
        
        # 条件注入
        self.condition_proj = nn.Linear(dim, dim)
    
    def forward(self, z_t, condition, t):
        """
        Args:
            z_t: 当前时间步的噪声样本
            condition: 条件特征 (RVQ 输出)
            t: 时间步 (0-1)
        """
        # 注入条件
        cond_emb = self.condition_proj(condition)
        
        # DiT 前向传播
        v_theta = self.dit(z_t, cond_emb, t)
        
        return v_theta
    
    @torch.no_grad()
    def decode(self, condition, num_steps=50):
        """
        从噪声重建潜在表示
        """
        # 初始化噪声
        z = torch.randn_like(condition)
        
        # 逐步去噪
        for t in reversed(range(num_steps)):
            t_normalized = t / num_steps
            
            # 预测向量场
            v = self(z, condition, t_normalized)
            
            # 更新 z
            dt = 1 / num_steps
            z = z + v * dt
        
        return z
```

### 2. DiT 架构

```python
class DiffusionTransformer(nn.Module):
    def __init__(self, dim, num_layers, num_heads, head_dim):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            DiTBlock(dim, num_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(dim)
        self.time_embed = timestep_embedding(dim)
    
    def forward(self, x, condition, t):
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 注入条件
        x = x + condition
        
        # 逐层处理
        for layer in self.layers:
            x = layer(x, t_emb)
        
        return self.norm_final(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)
    
    def forward(self, x, t_emb):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x) + t_emb)
        return x
```

### 3. 训练目标

```python
class FlowMatchingLoss:
    def __init__(self):
        self.num_steps = 50
    
    def compute_loss(self, model, condition, target_latent):
        """
        计算 Flow-Matching 损失
        """
        batch_size = condition.shape[0]
        
        # 随机时间步
        t = torch.rand(batch_size)
        
        # 噪声采样
        z_0 = torch.randn_like(target_latent)
        
        # 线性插值：z_t = t * z_1 + (1-t) * z_0
        z_t = t[:, None, None] * target_latent + (1 - t[:, None, None]) * z_0
        
        # 目标向量场：v = z_1 - z_0
        v_target = target_latent - z_0
        
        # 预测向量场
        v_pred = model(z_t, condition, t)
        
        # MSE 损失
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
```

### 4. Reflow 蒸馏

将采样步骤从 50 步减少到 10 步：

```python
class ReflowDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model  # 冻结
        self.student = student_model
    
    def distillation_loss(self, condition):
        # 教师模型生成轨迹
        with torch.no_grad():
            z_0 = torch.randn_like(condition)
            trajectories = []
            
            for t in np.linspace(0, 1, 50):
                v = self.teacher(z_t, condition, t)
                z_t = z_t + v * (1/50)
                trajectories.append(z_t)
            
            # 最终目标
            z_1 = trajectories[-1]
        
        # 学生模型学习教师轨迹
        loss = 0
        for i, t in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            z_t = trajectories[int(t * 50)]
            z_0_traj = trajectories[0]
            
            v_student = self.student(z_t, condition, t)
            v_teacher = (z_1 - z_0_traj)  # 简化的教师轨迹
            
            loss += F.mse_loss(v_student, v_teacher)
        
        return loss / 5
```

### 5. 解码器配置

```python
class AudioDecoder:
    def __init__(self):
        # Mel-VAE 解码器
        self.mel_vae = load_pretrained_mel_vae()
        
        # HiFi-GAN 声码器
        self.hifigan = load_pretrained_hifigan()
    
    @torch.no_grad()
    def decode(self, mel_features):
        """
        Args:
            mel_features: Mel-VAE 特征 [B, T, D]
        
        Returns:
            audio: 重建音频 [B, samples]
        """
        # Mel 频谱重建
        mel = self.mel_vae.decode(mel_features)
        
        # 波形生成
        audio = self.hifigan(mel)
        
        return audio
```

---

## 训练策略

### 1. 三阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     三阶段训练流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  阶段 1: 预训练                                                  │
│  ━━━━━━━━━━━━━━━━                                                │
│  • 数据：600,000+ 首歌曲                                         │
│  • 片段长度：20.48 秒                                            │
│  • GPU：88 × A100                                               │
│  • 批量：160                                                     │
│  • 轮数：15 epochs                                              │
│  • 学习率：1×10⁻⁴                                               │
│         │                                                        │
│         ▼                                                        │
│  阶段 2: 微调                                                    │
│  ━━━━━━━━━━━━━━━━                                                │
│  • 数据：50,000 高质量片段                                       │
│  • 片段长度：29.76 秒                                            │
│  • 目的：提升重建质量                                            │
│         │                                                        │
│         ▼                                                        │
│  阶段 3: Reflow 蒸馏                                             │
│  ━━━━━━━━━━━━━━━━                                                │
│  • 数据：20,000 超高质量样本                                     │
│  • 步骤：50 → 10                                                 │
│  • 目的：加速推理                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 学习率调度

```python
class CosineWarmupScheduler:
    def __init__(self, warmup_steps, total_steps, base_lr, min_lr):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            # Warmup 阶段
            return self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine 衰减
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
```

### 3. 损失函数组合

```python
class TotalLoss:
    def __init__(self):
        self.lambda_fm = 1.0      # Flow-Matching 损失
        self.lambda_commit = 1.0  # 承诺损失
        self.lambda_sem = 0.1     # 语义对齐损失
        self.lambda_pho = 0.1     # 语音对齐损失
    
    def compute(self, outputs, targets):
        loss_fm = self.lambda_fm * outputs['fm_loss']
        loss_commit = self.lambda_commit * outputs['commit_loss']
        loss_align_sem = self.lambda_sem * outputs['semantic_alignment_loss']
        loss_align_pho = self.lambda_pho * outputs['phonetic_alignment_loss']
        
        return loss_fm + loss_commit + loss_align_sem + loss_align_pho
```

### 4. 优化器配置

```python
def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    # 分层学习率
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': lr},
        {'params': model.quantizer.parameters(), 'lr': lr},
        {'params': model.decoder.parameters(), 'lr': lr * 0.1},  # 解码器用较小学习率
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    return optimizer
```

---

## 数据准备与处理

### 1. 数据集要求

| 指标 | 最低要求 | 推荐配置 |
|------|---------|----------|
| 歌曲数量 | 10,000 首 | 100,000+ 首 |
| 总时长 | 10,000 小时 | 100,000+ 小时 |
| 采样率 | 32 kHz | 48 kHz |
| 格式 | 单声道 | 立体声 |
| 质量 | 128 kbps | 320 kbps |

### 2. 数据预处理

```python
class AudioPreprocessor:
    def __init__(self, sample_rate=48000, segment_length=29.76):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
    
    def process_dataset(self, audio_files, output_dir):
        for audio_path in audio_files:
            # 1. 加载音频
            audio, sr = self.load_audio(audio_path)
            
            # 2. 重采样
            audio = self.resample(audio, sr)
            
            # 3. 归一化
            audio = self.normalize(audio)
            
            # 4. 分割为固定长度片段
            segments = self.segment(audio)
            
            # 5. 保存
            for i, segment in enumerate(segments):
                self.save_segment(segment, output_dir, audio_path, i)
    
    def load_audio(self, path):
        audio, sr = torchaudio.load(path)
        return audio, sr
    
    def resample(self, audio, sr):
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        return audio
    
    def normalize(self, audio):
        return audio / (torch.abs(audio).max() + 1e-8)
    
    def segment(self, audio):
        """分割为固定长度片段"""
        segments = []
        num_segments = len(audio) // self.segment_samples
        
        for i in range(num_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = audio[:, start:end]
            segments.append(segment)
        
        return segments
```

### 3. 特征提取

```python
class FeatureExtractor:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        
        # Mel 频谱参数
        self.n_fft = 2048
        self.hop_length = 480  # 10ms at 48kHz
        self.n_mels = 128
        
        # CQT 参数
        self.cqt_bins = 84
        self.cqt_fmin = 32.7  # C2
    
    def extract_mel(self, audio):
        """提取 Mel 频谱"""
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel = mel_fn(audio)
        return torch.log(mel + 1e-8)
    
    def extract_cqt(self, audio):
        """提取 CQT 特征"""
        cqt_fn = torchaudio.transforms.CQT(
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.cqt_bins,
            fmin=self.cqt_fmin
        )
        cqt = cqt_fn(audio)
        return torch.log(cqt + 1e-8)
```

### 4. 数据质量筛选

```python
class QualityFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def filter_dataset(self, dataset):
        """使用客观指标筛选高质量样本"""
        scores = []
        
        for audio in dataset:
            # 计算各种质量指标
            metrics = self.compute_metrics(audio)
            scores.append(metrics)
        
        # 筛选高质量样本
        filtered_dataset = [
            sample for sample, score in zip(dataset, scores)
            if self.quality_score(score) > self.threshold
        ]
        
        return filtered_dataset
    
    def compute_metrics(self, audio):
        """计算质量指标"""
        # FAD (Fréchet Audio Distance)
        fad = self.compute_fad(audio)
        
        # ViSQOL
        visqol = self.compute_visqol(audio)
        
        # 响度
        loudness = self.compute_loudness(audio)
        
        return {'fad': fad, 'visqol': visqol, 'loudness': loudness}
    
    def quality_score(self, metrics):
        """综合质量分数"""
        # 归一化并加权
        score = (1.0 - min(metrics['fad'], 10) / 10) * 0.4
        score += min(metrics['visqol'], 5) / 5 * 0.4
        score += min(metrics['loudness'], -5) / -5 * 0.2
        return score
```

---

## 训练流程详解

### 1. 完整训练脚本

```python
#!/usr/bin/env python3
"""
Music Tokenizer 完整训练脚本
基于 HeartMuLa 和 MuCodec 技术
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import argparse
import os


class MusicTokenizerTrainer:
    def __init__(self, config):
        self.config = config
        
        # 模型
        self.encoder = MuEncoder(config.encoder_dim, config.num_layers)
        self.quantizer = ResidualVectorQuantizer(
            config.encoder_dim,
            config.num_quantizers,
            config.codebook_size
        )
        self.decoder = FlowMatchingDecoder(
            config.decoder_dim,
            config.num_decoder_layers
        )
        
        # 优化器
        self.optimizer = self.create_optimizer()
        
        # 学习率调度
        self.scheduler = self.create_scheduler()
        
        # 混合精度
        self.scaler = GradScaler()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.quantizer.to(self.device)
        self.decoder.to(self.device)
    
    def create_optimizer(self):
        param_groups = [
            {'params': self.encoder.parameters(), 'lr': self.config.lr},
            {'params': self.quantizer.parameters(), 'lr': self.config.lr},
            {'params': self.decoder.parameters(), 'lr': self.config.lr * 0.1},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    def create_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.min_lr
        )
    
    def train_epoch(self, dataloader):
        self.encoder.train()
        self.quantizer.train()
        self.decoder.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            audio = batch['audio'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # 1. 编码
                features = self.encoder(audio)
                
                # 2. 量化
                quantized, commit_loss, indices = self.quantizer(features)
                
                # 3. 重建
                target = self.get_target_latent(audio)  # 需要预处理
                fm_loss = self.compute_fm_loss(quantized, target)
                
                # 4. 总损失
                loss = fm_loss + commit_loss * self.config.commit_weight
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """评估模型"""
        self.encoder.eval()
        self.quantizer.eval()
        self.decoder.eval()
        
        metrics = {
            'visqol': 0,
            'fad': 0,
            'stoi': 0,
            'pesq': 0,
        }
        
        for batch in dataloader:
            audio = batch['audio'].to(self.device)
            
            # 前向传播
            features = self.encoder(audio)
            quantized, _, _ = self.quantizer(features)
            reconstructed = self.decoder.decode(quantized)
            
            # 计算指标
            batch_metrics = self.compute_metrics(audio, reconstructed)
            for key in metrics:
                metrics[key] += batch_metrics[key]
        
        # 平均
        for key in metrics:
            metrics[key] /= len(dataloader)
        
        return metrics
    
    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        best_metric = float('-inf')
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 评估
            val_metrics = self.evaluate(val_dataset)
            
            # 学习率更新
            self.scheduler.step()
            
            # 日志
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val ViSQOL: {val_metrics['visqol']:.4f}")
            print(f"  Val FAD: {val_metrics['fad']:.4f}")
            
            # 保存最佳模型
            if val_metrics['visqol'] > best_metric:
                best_metric = val_metrics['visqol']
                self.save_checkpoint('best_model.pt')
            
            # 定期保存
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
    
    def save_checkpoint(self, path):
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, path))


# 配置
class Config:
    # 模型配置
    encoder_dim = 512
    num_layers = 13
    num_quantizers = 8
    codebook_size = 8192
    decoder_dim = 512
    num_decoder_layers = 24
    
    # 训练配置
    batch_size = 16
    num_epochs = 100
    lr = 1e-4
    min_lr = 1e-6
    commit_weight = 1.0
    
    # 其他
    checkpoint_dir = 'checkpoints'
    save_interval = 10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # 创建训练器
    trainer = MusicTokenizerTrainer(Config)
    
    # 加载数据
    train_dataset = AudioDataset(args.data_dir, split='train')
    val_dataset = AudioDataset(args.data_dir, split='val')
    
    # 开始训练
    trainer.train(train_dataset, val_dataset)
```

### 2. 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())


def create_distributed_trainer(config):
    setup_distributed()
    
    # 创建模型
    model = MusicTokenizerTrainer(config)
    
    # DDP 包装
    model.encoder = DDP(model.encoder, device_ids=[local_rank])
    model.quantizer = DDP(model.quantizer, device_ids=[local_rank])
    model.decoder = DDP(model.decoder, device_ids=[local_rank])
    
    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset)
    
    return model, train_sampler
```

### 3. 推理脚本

```python
@torch.no_grad()
def encode_audio(tokenizer, audio):
    """编码音频为 token"""
    tokenizer.eval()
    audio = audio.unsqueeze(0).to(tokenizer.device)
    
    features = tokenizer.encoder(audio)
    _, _, indices = tokenizer.quantizer(features)
    
    return indices  # 返回 token 序列


@torch.no_grad()
def decode_tokens(tokenizer, indices):
    """从 token 重建音频"""
    tokenizer.eval()
    
    # 重建量化表示
    quantized = tokenizer.quantizer.decode(indices)
    
    # 重建音频
    audio = tokenizer.decoder.decode(quantized)
    
    return audio.squeeze(0).cpu()


# 使用示例
tokenizer = load_pretrained_tokenizer('checkpoints/best_model.pt')

# 编码
audio = load_audio('test_audio.wav')
tokens = encode_audio(tokenizer, audio)

# 解码
reconstructed = decode_tokens(tokenizer, tokens)

# 保存
torchaudio.save('reconstructed.wav', reconstructed, 48000)
```

---

## 评估指标

### 1. 客观指标

| 指标 | 描述 | 方向 | 阈值 |
|------|------|------|------|
| **VISQOL** | 语音/音频质量客观度量 | ↑ | > 3.5 |
| **FAD** | Fréchet Audio Distance | ↓ | < 1.0 |
| **FD** | Fréchet Distance | ↓ | < 15 |
| **STOI** | 短时客观可懂度 | ↑ | > 0.6 |
| **PESQ** | 感知语音质量评估 | ↑ | > 1.5 |
| **SPK_SIM** | 说话人相似度 | ↑ | > 0.85 |
| **WER** | 词错误率 | ↓ | < 0.3 |

### 2. 计算代码

```python
class Evaluator:
    def __init__(self):
        # 加载预训练模型
        self.stoi_model = load_stoi_model()
        self.pesq_model = load_pesq_model()
        self.spk_sim_model = load_spk_sim_model()
    
    def compute_all_metrics(self, reference, generated):
        """计算所有指标"""
        metrics = {}
        
        # VISQOL
        metrics['visqol'] = self.compute_visqol(reference, generated)
        
        # FAD
        metrics['fad'] = self.compute_fad(reference, generated)
        
        # STOI
        metrics['stoi'] = self.compute_stoi(reference, generated)
        
        # PESQ
        metrics['pesq'] = self.compute_pesq(reference, generated)
        
        # 说话人相似度
