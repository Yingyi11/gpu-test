# DDP vs FSDP 性能对比报告

## 📊 性能对比总结

### 测试配置
- **硬件**: 8x GPU
- **模型**: ResNet50
- **Batch Size**: 128 per GPU (全局 1024)
- **迭代次数**: 100
- **Workers**: 12

---

## 🏆 性能排行榜

| 方法 | 总吞吐量 | 每卡吞吐量 | 显存使用 | 相对性能 |
|------|---------|-----------|---------|---------|
| **DDP + FP32** | **4629.5 img/s** | **578.7 img/s** | 13.0 GB | **100%** ⭐ |
| **FSDP + FP16** | 3183.4 img/s | 397.9 img/s | **8.5 GB** | 68.8% |
| **FSDP + FP32** | 2516.2 img/s | 314.5 img/s | 12.8 GB | 54.4% |

---

## 📈 详细性能数据

### 1. DDP (DistributedDataParallel) - FP32
```
方法: 标准数据并行
精度: FP32
策略: 每个 GPU 保留完整模型副本

性能:
- 总吞吐量: 4629.5 images/s
- 每 GPU: 578.7 images/s  
- 迭代时间: 222.36ms (平均)
- 扩展效率: 76.9% (vs 单卡)

显存:
- 已分配: 12.99 GB
- 已预留: 13.16 GB

优点: ✅ 最快的吞吐量
缺点: ❌ 每个 GPU 需要完整模型副本
```

### 2. FSDP (Fully Sharded Data Parallel) - FP16
```
方法: 完全分片数据并行 + 混合精度
精度: FP16
策略: Sharding策略=FULL_SHARD

性能:
- 总吞吐量: 3183.4 images/s (DDP 的 68.8%)
- 每 GPU: 397.9 images/s
- 迭代时间: 323.40ms (平均)

显存:
- 已分配: 8.47 GB (比 DDP 节省 35%)
- 已预留: 8.97 GB

优点: ✅ 显著节省显存 (35%)
      ✅ 可训练更大模型
      ✅ 比 FP32 快 26%
缺点: ❌ 通信开销较大
      ❌ 比 DDP 慢 31%
```

### 3. FSDP (Fully Sharded Data Parallel) - FP32
```
方法: 完全分片数据并行
精度: FP32
策略: Sharding策略=FULL_SHARD

性能:
- 总吞吐量: 2516.2 images/s (DDP 的 54.4%)
- 每 GPU: 314.5 images/s
- 迭代时间: 407.38ms (平均)

显存:
- 已分配: 12.82 GB
- 已预留: 13.00 GB

优点: ✅ 与 DDP 显存使用相当
缺点: ❌ 最慢 (比 DDP 慢 45.6%)
      ❌ 通信开销最大
```

---

## 🎯 关键发现

### 1. **DDP 性能最优** ⭐
- 对于 ResNet50 这种中等大小模型，DDP 是最佳选择
- 通信开销小，扩展效率高 (76.9%)
- 适合显存充足的场景

### 2. **FSDP 显存效率高** 💾
- FP16 模式节省 **35% 显存** (8.5GB vs 13.0GB)
- 适合训练大模型 (如 LLaMA, GPT)
- 可以用更大的 batch size

### 3. **FSDP 通信开销大** ⚠️
- 完全分片导致每次前后向传播都需要 all-gather/reduce-scatter
- FP32 模式性能损失 **45.6%**
- FP16 模式性能损失 **31.2%**

### 4. **混合精度加速明显** 🚀
- FSDP FP16 比 FP32 快 **26.5%** (3183.4 vs 2516.2 img/s)
- 显存使用相同
- 强烈推荐 FSDP 使用 FP16

---

## 💡 使用建议

### 选择 DDP 的场景:
✅ 模型可以装入单卡显存  
✅ 追求最高训练速度  
✅ 显存充足 (>16GB)  
✅ 模型参数量 < 10B

**推荐命令:**
```bash
uv run torchrun --nproc_per_node=8 \
    benchmark_resnet50.py \
    --batch-size 128 \
    --workers 12
```

### 选择 FSDP 的场景:
✅ 模型太大无法装入单卡  
✅ 需要节省显存  
✅ 训练超大模型 (>10B 参数)  
✅ 可以接受一定性能损失

**推荐命令 (FSDP + FP16):**
```bash
uv run torchrun --nproc_per_node=8 \
    benchmark_resnet50_fsdp.py \
    --batch-size 128 \
    --workers 12 \
    --amp
```

### 选择 FSDP + CPU Offload 的场景:
✅ 极大模型 (>100B 参数)  
✅ 显存严重不足  
✅ 可以接受更大性能损失

---

## 🔬 深入分析

### 为什么 FSDP 更慢?

1. **通信模式差异**:
   - **DDP**: 只在反向传播后同步梯度 (1次 all-reduce)
   - **FSDP**: 前向传播 all-gather 参数 + 反向传播 reduce-scatter 梯度 (多次)

2. **计算与通信重叠**:
   - **DDP**: 梯度同步可以与计算重叠
   - **FSDP**: all-gather 必须在计算前完成,阻塞更明显

3. **通信量对比** (每次迭代):
   ```
   DDP 通信量  = 模型参数大小 × 1
   FSDP 通信量 = 模型参数大小 × 2-3 (all-gather + reduce-scatter)
   ```

### 为什么 FSDP + FP16 更快?

1. **通信量减半**: FP16 参数大小是 FP32 的一半
2. **计算加速**: Tensor Core 对 FP16 有硬件加速
3. **带宽利用**: 相同带宽可以传输 2 倍数据

---

## 📊 性能曲线对比

### 吞吐量对比 (images/s)
```
5000 |                    ● DDP FP32 (4629.5)
4000 |              
3000 |          ● FSDP FP16 (3183.4)
2000 |     
1000 |  ● FSDP FP32 (2516.2)
   0 +----------------------------------------
       FSDP-FP32   FSDP-FP16    DDP-FP32
```

### 显存使用对比 (GB)
```
14 |  ● DDP (13.0 GB)       ● FSDP-FP32 (12.8 GB)
12 |  
10 |  
 8 |             ● FSDP-FP16 (8.5 GB)
 6 |  
 4 |  
 2 |  
 0 +----------------------------------------
     FSDP-FP16   FSDP-FP32    DDP-FP32
```

---

## 🎓 高级优化建议

### 1. FSDP Sharding 策略选择
```python
# FULL_SHARD: 完全分片 (最省显存,最慢)
sharding_strategy = ShardingStrategy.FULL_SHARD

# HYBRID_SHARD: 节点内复制,节点间分片 (平衡)
sharding_strategy = ShardingStrategy.HYBRID_SHARD

# SHARD_GRAD_OP: 只分片梯度和优化器 (类似 ZeRO-2,更快)
sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

# NO_SHARD: 不分片 (等同于 DDP)
sharding_strategy = ShardingStrategy.NO_SHARD
```

### 2. 测试其他策略
```bash
# ZeRO-2 风格 (只分片梯度和优化器)
uv run torchrun --nproc_per_node=8 \
    benchmark_resnet50_fsdp.py \
    --batch-size 128 \
    --fsdp-strategy zero2 \
    --amp

# 混合分片 (单节点内不分片)
uv run torchrun --nproc_per_node=8 \
    benchmark_resnet50_fsdp.py \
    --batch-size 128 \
    --fsdp-strategy hybrid \
    --amp
```

### 3. 增大 Batch Size
由于 FSDP FP16 只用了 8.5GB 显存，可以增大 batch size:
```bash
uv run torchrun --nproc_per_node=8 \
    benchmark_resnet50_fsdp.py \
    --batch-size 192 \  # or 256
    --amp
```

---

## 📝 总结

### 最佳实践矩阵

| 模型大小 | 显存情况 | 推荐方案 | 预期性能 |
|---------|---------|---------|---------|
| < 1B 参数 | 充足 (>16GB) | **DDP** | 最快 |
| 1B-10B | 充足 | **DDP** | 最快 |
| 1B-10B | 紧张 (<16GB) | **FSDP + FP16** | 中等 |
| 10B-100B | 紧张 | **FSDP + FP16** | 中等 |
| > 100B | 极度紧张 | **FSDP + FP16 + CPU Offload** | 较慢 |

### 关键要点

1. ⭐ **对于 ResNet50**: DDP 是最佳选择 (快 54%)
2. 💾 **显存受限**: 使用 FSDP + FP16 (节省 35% 显存)
3. 🚀 **FSDP 必开 FP16**: 性能提升 26%,无额外成本
4. 📊 **扩展效率**: DDP 76.9% > FSDP FP16 52.9% > FSDP FP32 41.8%

---

**测试时间**: 2025-11-11  
**PyTorch 版本**: 2.9.0+cu128  
**CUDA 版本**: 12.8  
**测试硬件**: 8x GPU
