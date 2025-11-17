# GPU 显存最大化 + SwanLab 监控指南

## 功能特点

本增强版测试脚本提供以下功能：

### 1. 显存最大化
- **更大的 Batch Size**: 从 128 增加到 256（全局 batch size 从 1024 增加到 2048）
- **显存预分配**: 可以指定目标显存使用量（默认 20GB/GPU）
- **自动填充**: 自动分配额外的显存缓冲区以达到目标使用量

### 2. SwanLab 实时监控
集成了 SwanLab 平台，实时监控和上传以下指标：

#### 训练指标
- `train/loss`: 训练损失
- `train/throughput`: 吞吐量（images/s）
- `train/iter_time_ms`: 每次迭代时间（毫秒）
- `train/step`: 训练步数

#### GPU 指标（每个 GPU）
- `gpu_{i}/utilization`: GPU 利用率 (%)
- `gpu_{i}/memory_used_mb`: 已使用显存 (MB)
- `gpu_{i}/memory_util`: 显存利用率 (%)
- `gpu_{i}/temperature`: GPU 温度 (°C)
- `gpu_{i}/power_w`: 功耗 (W)

#### 平均 GPU 指标
- `gpu_avg/utilization`: 平均 GPU 利用率
- `gpu_avg/memory_used_gb`: 总显存使用量 (GB)
- `gpu_avg/memory_util`: 平均显存利用率
- `gpu_avg/temperature_max`: 最高温度
- `gpu_avg/power_total_w`: 总功耗 (W)

#### 总结指标
- `summary/avg_throughput`: 平均吞吐量
- `summary/avg_iter_time_ms`: 平均迭代时间
- `summary/total_time_s`: 总运行时间
- `summary/max_memory_allocated_gb`: 最大显存分配
- `summary/max_memory_reserved_gb`: 最大显存预留

### 3. 后台监控线程
- 独立的监控线程每 5 秒采集一次 GPU 指标
- 不影响训练性能
- 自动在训练开始时启动，结束时停止

## 使用方法

### 快速测试（推荐先运行）

```bash
./test_maxmem_swanlab.sh
```

这将运行 50 次迭代的快速测试，验证：
- 显存是否正确分配
- SwanLab 是否正常上传
- 多卡训练是否稳定

### 完整 15 分钟压力测试

```bash
./stress_test_15min.sh
```

这将运行约 15 分钟的完整压力测试，包括：
- 2300 次迭代
- Batch size 256/GPU（全局 2048）
- 目标显存 20GB/GPU
- 完整的 SwanLab 监控

### 自定义运行

```bash
uv run torchrun \
    --nproc_per_node=8 \
    --master_port=29510 \
    benchmark_resnet50_maxmem.py \
    --batch-size 256 \
    --iterations 1000 \
    --target-memory 20 \
    --enable-swanlab \
    --swanlab-project "my-project"
```

## 命令行参数

### 模型参数
- `--model`: 模型名称（resnet50/resnet101/resnet152，默认 resnet50）

### 训练参数
- `--batch-size`: 每 GPU 的批次大小（默认 256）
- `--iterations`: 测试迭代次数（默认 100）
- `--warmup`: 预热迭代次数（默认 10）
- `--workers`: 数据加载线程数（默认 12）

### 优化器参数
- `--optimizer`: 优化器类型（sgd/adam/adamw，默认 sgd）
- `--lr`: 学习率（默认 0.1）
- `--amp`: 启用混合精度训练

### 显存管理
- `--target-memory`: 目标显存使用量（GB），0 表示不预分配（默认 0）

### SwanLab 监控
- `--enable-swanlab`: 启用 SwanLab 监控
- `--swanlab-project`: SwanLab 项目名称（默认 'gpu-benchmark'）
- `--monitor-interval`: GPU 监控采样间隔（秒，默认 5）

## SwanLab 设置

### 首次使用

1. 注册 SwanLab 账号：访问 https://swanlab.cn

2. 获取 API Key：
   - 登录后进入设置页面
   - 复制你的 API Key

3. 配置本地环境：
```bash
export SWANLAB_API_KEY="your-api-key-here"
```

或者在首次运行时，SwanLab 会提示你登录。

### 查看监控数据

1. 访问 https://swanlab.cn
2. 进入你的项目（如 "gpu-stress-test"）
3. 查看实时监控图表和指标

## 性能优化建议

### 最大化显存使用

1. **增大 Batch Size**:
   ```bash
   --batch-size 256  # 或更大，直到 OOM
   ```

2. **预分配显存**:
   ```bash
   --target-memory 22  # 根据你的 GPU 容量调整
   ```

3. **使用更大的模型**:
   ```bash
   --model resnet152  # ResNet152 比 ResNet50 大 3 倍
   ```

### 示例：24GB GPU 最大化配置

```bash
uv run torchrun \
    --nproc_per_node=8 \
    benchmark_resnet50_maxmem.py \
    --model resnet152 \
    --batch-size 192 \
    --target-memory 22 \
    --enable-swanlab
```

### 示例：16GB GPU 配置

```bash
uv run torchrun \
    --nproc_per_node=8 \
    benchmark_resnet50_maxmem.py \
    --batch-size 128 \
    --target-memory 14 \
    --enable-swanlab
```

## 监控指标说明

### GPU 利用率
- **目标**: 95-100%
- **如果过低**: 增大 batch size 或减少 workers

### 显存利用率
- **目标**: 80-95%（留一些余量）
- **如果过低**: 增大 batch size 或 target-memory
- **如果 OOM**: 减小 batch size 或 target-memory

### 温度
- **正常范围**: 60-85°C
- **如果过高**: 检查散热系统

### 功耗
- **正常范围**: 接近 GPU 的 TDP
- **如果过低**: GPU 可能未充分利用

### 吞吐量
- **监控趋势**: 应该保持稳定
- **如果下降**: 可能有热节流或其他问题

## 故障排查

### SwanLab 连接失败
1. 检查 API Key 是否正确
2. 检查网络连接
3. 查看 SwanLab 日志：`~/.swanlab/logs`

### 显存不足 (OOM)
1. 减小 `--batch-size`
2. 减小 `--target-memory`
3. 使用较小的模型（resnet50 而不是 resnet152）

### GPU 利用率低
1. 增大 `--batch-size`
2. 减少 `--workers`（数据加载不应成为瓶颈）
3. 检查数据加载是否成为瓶颈

### 监控数据未上传
1. 确认使用了 `--enable-swanlab` 参数
2. 检查 SwanLab API Key 配置
3. 查看终端输出的错误信息

## 性能基准

### 预期性能（8x GPU）

**配置**: ResNet50, Batch Size 256/GPU, FP32

| 指标 | 预期值 |
|------|--------|
| 吞吐量 | ~3000-4000 images/s |
| 每 GPU 吞吐量 | ~400-500 images/s |
| 迭代时间 | ~500-700ms |
| GPU 利用率 | 95-100% |
| 显存使用 | 18-22 GB/GPU |

**配置**: ResNet152, Batch Size 128/GPU, FP32

| 指标 | 预期值 |
|------|--------|
| 吞吐量 | ~1000-1500 images/s |
| 每 GPU 吞吐量 | ~125-200 images/s |
| 迭代时间 | ~700-1000ms |
| GPU 利用率 | 95-100% |
| 显存使用 | 20-23 GB/GPU |

## 文件说明

- `benchmark_resnet50_maxmem.py`: 增强版基准测试脚本（显存最大化 + SwanLab）
- `test_maxmem_swanlab.sh`: 快速功能测试脚本（50 次迭代）
- `stress_test_15min.sh`: 15 分钟完整压力测试脚本（已更新）
- `MAXMEM_SWANLAB_GUIDE.md`: 本文档

## 注意事项

1. **首次运行**: 建议先运行 `test_maxmem_swanlab.sh` 验证配置
2. **显存安全**: 从较小的 target-memory 开始，逐步增加
3. **监控采样**: 默认 5 秒间隔，可根据需要调整
4. **网络连接**: SwanLab 需要互联网连接才能上传数据
5. **API 限制**: 注意 SwanLab 的 API 调用限制

## 下一步

1. ✅ 已安装 SwanLab 和 pynvml
2. ✅ 已创建增强版测试脚本
3. ✅ 已集成 GPU 监控
4. ⬜ 运行快速测试验证功能
5. ⬜ 配置 SwanLab API Key
6. ⬜ 运行完整 15 分钟压力测试
7. ⬜ 在 SwanLab 平台查看监控数据

运行快速测试：
```bash
./test_maxmem_swanlab.sh
```
