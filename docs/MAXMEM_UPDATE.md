# 显存最大化 + SwanLab 监控更新

## 🎉 新功能

已成功实现以下两个主要目标：

### 1. ✅ 显存最大化
- **更大的 Batch Size**: 256/GPU（全局 2048），是之前的 2 倍
- **显存预分配**: 可指定目标显存使用量（如 20GB/GPU）
- **更大模型支持**: ResNet50/101/152 可选
- **预期显存使用**: 18-22 GB/GPU（满载）

### 2. ✅ SwanLab 实时监控
- **训练指标**: Loss, Throughput, 迭代时间
- **GPU 指标**: 利用率、显存、温度、功耗（每个 GPU）
- **平均指标**: 所有 GPU 的平均/总计指标
- **自动上传**: 后台监控线程每 5 秒采集并上传
- **实时可视化**: 在 SwanLab 平台查看实时图表

## 📁 新文件

1. **`benchmark_resnet50_maxmem.py`** - 增强版基准测试脚本
   - 集成显存最大化功能
   - 集成 SwanLab 监控
   - 后台 GPU 监控线程
   - 支持所有原有功能

2. **`test_maxmem_swanlab.sh`** - 快速功能测试
   - 50 次迭代快速验证
   - 测试显存分配和 SwanLab 上传
   - 建议首次运行

3. **`MAXMEM_SWANLAB_GUIDE.md`** - 详细使用指南
   - 完整的功能说明
   - 使用示例和参数说明
   - SwanLab 配置指南
   - 故障排查

## 🚀 快速开始

### 步骤 1: 配置 SwanLab

```bash
# 访问 https://swanlab.cn 注册并获取 API Key
export SWANLAB_API_KEY="your-api-key-here"
```

### 步骤 2: 运行快速测试

```bash
./test_maxmem_swanlab.sh
```

这将验证：
- ✅ 显存是否正确分配到 ~20GB/GPU
- ✅ GPU 利用率是否达到 95%+
- ✅ SwanLab 是否正常上传数据

### 步骤 3: 查看监控数据

1. 访问 https://swanlab.cn
2. 进入项目 "gpu-stress-test"
3. 查看实时图表：
   - 训练 Loss 曲线
   - GPU 利用率趋势
   - 显存使用情况
   - 温度和功耗监控

### 步骤 4: 运行完整压力测试

```bash
./stress_test_15min.sh
```

这将运行 15 分钟的完整测试，持续监控并上传数据。

## 📊 监控指标一览

### 训练指标
- `train/loss` - 训练损失
- `train/throughput` - 吞吐量（images/s）
- `train/iter_time_ms` - 迭代时间

### GPU 指标（单卡）
- `gpu_0-7/utilization` - GPU 利用率
- `gpu_0-7/memory_used_mb` - 显存使用
- `gpu_0-7/temperature` - 温度
- `gpu_0-7/power_w` - 功耗

### 平均指标
- `gpu_avg/utilization` - 平均利用率
- `gpu_avg/memory_used_gb` - 总显存使用
- `gpu_avg/temperature_max` - 最高温度
- `gpu_avg/power_total_w` - 总功耗

## 🎯 预期性能

**配置**: 8x GPU, ResNet50, Batch Size 256/GPU

| 指标 | 目标值 |
|------|--------|
| GPU 利用率 | 95-100% ✅ |
| 显存使用 | 18-22 GB/GPU ✅ |
| 吞吐量 | ~3500 images/s |
| 温度 | 60-85°C |

## 🔧 自定义配置

### 最大化显存（24GB GPU）
```bash
uv run torchrun --nproc_per_node=8 benchmark_resnet50_maxmem.py \
    --model resnet152 \
    --batch-size 192 \
    --target-memory 22 \
    --enable-swanlab
```

### 16GB GPU 适配
```bash
uv run torchrun --nproc_per_node=8 benchmark_resnet50_maxmem.py \
    --batch-size 128 \
    --target-memory 14 \
    --enable-swanlab
```

### 混合精度（FP16）
```bash
uv run torchrun --nproc_per_node=8 benchmark_resnet50_maxmem.py \
    --batch-size 384 \
    --target-memory 22 \
    --amp \
    --enable-swanlab
```

## 📈 SwanLab 优势

1. **实时可视化**: 无需等待训练结束
2. **多指标对比**: 同时查看多个 GPU 的指标
3. **历史记录**: 保存所有实验数据
4. **远程监控**: 随时随地通过网页查看
5. **警报功能**: 可设置阈值警报

## 🔍 与原版对比

| 特性 | 原版 | 新版 (maxmem) |
|------|------|---------------|
| Batch Size | 128/GPU | 256/GPU |
| 全局 Batch | 1024 | 2048 |
| 显存使用 | ~10-12 GB | ~18-22 GB |
| 监控方式 | 终端输出 | SwanLab 实时上传 |
| 监控指标 | 训练指标 | 训练 + GPU 硬件指标 |
| 可视化 | 无 | 实时图表 |
| 历史记录 | 无 | 永久保存 |

## 📚 相关文档

- **MAXMEM_SWANLAB_GUIDE.md** - 完整使用指南
- **PERFORMANCE_SUMMARY.md** - 原始性能测试结果
- **DDP_vs_FSDP_COMPARISON.md** - DDP vs FSDP 对比

## ⚠️ 注意事项

1. **首次使用**: 请先运行 `test_maxmem_swanlab.sh` 验证配置
2. **显存安全**: 从较小的 target-memory 开始，逐步增加
3. **网络需求**: SwanLab 需要互联网连接
4. **API Key**: 需要在 SwanLab 注册并配置 API Key

## 🐛 故障排查

### SwanLab 无法连接
```bash
# 检查 API Key
echo $SWANLAB_API_KEY

# 重新登录
swanlab login
```

### 显存不足 (OOM)
```bash
# 减小 batch size
--batch-size 192  # 或更小

# 减小目标显存
--target-memory 16  # 或更小
```

### GPU 利用率低
```bash
# 增大 batch size
--batch-size 320

# 使用更大的模型
--model resnet152
```

## 🎊 完成状态

- ✅ 安装 SwanLab 和 pynvml
- ✅ 创建增强版基准测试脚本
- ✅ 集成 GPU 监控功能
- ✅ 更新 15 分钟压力测试脚本
- ✅ 创建快速测试脚本
- ✅ 编写完整使用指南

**下一步**: 运行 `./test_maxmem_swanlab.sh` 开始测试！
