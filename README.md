# 🚀 GPU 多卡训练测试工具套件

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一套完整的多卡 GPU 训练环境测试工具，用于验证和评估服务器的多 GPU 并行训练能力。支持环境检测、通信测试、性能基准测试等功能。

## 📋 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [工具说明](#️-工具说明)
- [性能基准](#-性能基准)
- [故障排查](#-故障排查)
- [常见问题](#-常见问题)

## ✨ 功能特性

- ✅ **自动化环境检测** - 一键检测驱动、CUDA、NCCL、PyTorch 环境
- ✅ **NCCL 通信测试** - 验证多卡通信能力和带宽性能
- ✅ **分布式训练测试** - DDP 和 FSDP 训练验证
- ✅ **性能基准测试** - ResNet50 真实模型吞吐量测试
- ✅ **GPU 监控工具** - 实时监控 GPU 使用情况
- ✅ **详细日志记录** - 自动生成测试报告和性能数据

## 🚀 快速开始

### 环境要求

- **硬件**: NVIDIA GPU × 1-8 (或更多)
- **驱动**: NVIDIA Driver >= 450.x
- **CUDA**: CUDA >= 11.0
- **Python**: Python >= 3.10
- **PyTorch**: PyTorch >= 2.0 (带 CUDA 支持)

### 安装依赖

```bash
# 使用 pip 安装
pip install torch torchvision numpy pandas matplotlib pynvml swanlab

# 或使用 uv (推荐)
uv sync
```

### 一键环境检测

```bash
# 赋予执行权限
chmod +x scripts/test_8gpu.sh

# 运行完整环境检测
bash scripts/test_8gpu.sh
```

测试完成后会生成详细日志文件 `gpu_test_YYYYMMDD_HHMMSS.log`。

## 📁 项目结构

```
gpu-test/
├── README.md                 # 项目说明文档（本文件）
├── pyproject.toml           # 项目配置和依赖管理
│
├── benchmarks/              # 性能基准测试
│   ├── benchmark_resnet50.py           # ResNet50 DDP 基准测试
│   ├── benchmark_resnet50_fsdp.py      # ResNet50 FSDP 基准测试
│   └── benchmark_resnet50_maxmem.py    # 最大显存利用率测试
│
├── tests/                   # 功能测试脚本
│   ├── test_nccl_quick.py              # NCCL 通信快速测试
│   └── test_ddp.py                     # DDP 训练功能测试
│
├── scripts/                 # 工具脚本
│   ├── test_8gpu.sh                    # 一键环境检测脚本
│   ├── monitor_gpu.sh                  # GPU 实时监控脚本
│   ├── stress_test_15min.sh            # 15分钟压力测试
│   ├── test_maxmem_swanlab.sh          # SwanLab 监控测试
│   └── plot_swanlab.py                 # SwanLab 数据可视化
│
└── docs/                    # 文档资料
    ├── DDP_vs_FSDP_COMPARISON.md       # DDP vs FSDP 对比分析
    ├── PERFORMANCE_SUMMARY.md          # 性能测试总结
    ├── MAXMEM_SWANLAB_GUIDE.md         # SwanLab 使用指南
    ├── MAXMEM_UPDATE.md                # 显存优化更新说明
    └── NCCL_SHM_DIAGNOSIS.md           # NCCL 共享内存诊断
```

## 🛠️ 工具说明

### 1. 环境检测 - `scripts/test_8gpu.sh`

全自动环境检测脚本，验证多卡训练所需的所有组件。

```bash
bash scripts/test_8gpu.sh
```

**检测内容:**
- 系统信息和 NVIDIA 驱动版本
- GPU 数量、型号和拓扑结构
- CUDA 和 cuDNN 版本
- PyTorch 环境和 NCCL 支持
- 分布式通信能力
- GPU 互联带宽

### 2. NCCL 通信测试 - `tests/test_nccl_quick.py`

快速测试 GPU 间通信性能，不涉及实际训练。

```bash
# 自动启动（推荐）
python tests/test_nccl_quick.py

# 使用 torchrun
torchrun --nproc_per_node=8 tests/test_nccl_quick.py
```

**测试操作:**
- all_reduce - 全局归约操作
- broadcast - 广播操作
- all_gather - 全局收集操作
- reduce_scatter - 归约分散操作
- 不同数据量的带宽测试 (1MB, 4MB, 16MB, 64MB)

### 3. DDP 训练测试 - `tests/test_ddp.py`

验证 PyTorch DistributedDataParallel 训练流程。

```bash
# 自动启动（推荐）
python tests/test_ddp.py

# 使用 torchrun
torchrun --nproc_per_node=8 tests/test_ddp.py
```

**测试阶段:**
- 第一阶段: NCCL 通信正确性验证
- 第二阶段: 简单神经网络训练
- 第三阶段: 梯度同步和多轮训练稳定性

### 4. ResNet50 基准测试 - `benchmarks/benchmark_resnet50.py`

使用真实模型测试训练吞吐量，更接近实际训练场景。

```bash
# 基础测试
python benchmarks/benchmark_resnet50.py

# 自定义配置
python benchmarks/benchmark_resnet50.py --batch-size 128 --iterations 200

# 使用混合精度加速
python benchmarks/benchmark_resnet50.py --amp --batch-size 256

# 使用 torchrun
torchrun --nproc_per_node=8 benchmarks/benchmark_resnet50.py
```

**主要参数:**
- `--model` - 模型类型 (resnet50/resnet101/resnet152)
- `--batch-size` - 每 GPU 的批次大小 (默认: 64)
- `--iterations` - 测试迭代次数 (默认: 100)
- `--amp` - 启用混合精度训练
- `--optimizer` - 优化器类型 (sgd/adam/adamw)

### 5. FSDP 基准测试 - `benchmarks/benchmark_resnet50_fsdp.py`

测试 Fully Sharded Data Parallel 性能，适合大模型训练。

```bash
torchrun --nproc_per_node=8 benchmarks/benchmark_resnet50_fsdp.py
```

FSDP 可以显著降低显存占用，适合训练超大模型。详见 `docs/DDP_vs_FSDP_COMPARISON.md`。

### 6. GPU 监控 - `scripts/monitor_gpu.sh`

实时监控 GPU 使用情况，适合长时间训练监控。

```bash
bash scripts/monitor_gpu.sh
```

每秒刷新显示：
- GPU 利用率
- 显存使用情况
- GPU 温度
- 功耗状态

### 7. 压力测试 - `scripts/stress_test_15min.sh`

15 分钟连续压力测试，验证长时间运行稳定性。

```bash
bash scripts/stress_test_15min.sh
```

## 📊 性能基准

### NCCL 带宽参考值

| GPU 型号 | 互联方式 | all_reduce (64MB) |
|---------|---------|-------------------|
| A100 80GB × 8 | NVLink 3.0 | 200-300 GB/s |
| A100 40GB × 8 | NVLink 3.0 | 200-300 GB/s |
| V100 32GB × 8 | NVLink 2.0 | 100-150 GB/s |
| RTX 4090 × 8 | PCIe 4.0 | 30-50 GB/s |
| RTX 3090 × 8 | PCIe 4.0 | 20-40 GB/s |

### ResNet50 训练吞吐量

| GPU 型号 | Batch Size | FP32 (images/s) | AMP (images/s) |
|---------|-----------|----------------|---------------|
| A100 80GB × 8 | 64 | 1800-2200 | 3500-4500 |
| A100 40GB × 8 | 64 | 1700-2100 | 3400-4300 |
| V100 32GB × 8 | 64 | 1200-1500 | 2400-3000 |
| RTX 4090 × 8 | 64 | 1400-1800 | 2800-3600 |
| RTX 3090 × 8 | 64 | 1000-1300 | 2000-2600 |

*注: 实际性能受多种因素影响（CPU、内存、存储、网络等）*

## 🔧 故障排查

### 1. GPU 检测不到或数量不足

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 Docker GPU 映射（容器内）
docker run --gpus all ...
```

### 2. NCCL 初始化失败

```bash
# 检查共享内存大小
df -h /dev/shm  # 应该 >= 8GB

# Docker 用户添加参数
docker run --ipc=host --gpus all ...

# 设置 NCCL 调试信息
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand
```

### 3. 通信带宽低于预期

```bash
# 查看 GPU 拓扑结构
nvidia-smi topo -m

# 检查 NVLink 状态
nvidia-smi nvlink --status

# 优化 NCCL 参数
export NCCL_ALGO=Ring
export NCCL_MIN_NRINGS=8
```

### 4. 显存不足 (OOM)

```bash
# 减小 batch size
python benchmarks/benchmark_resnet50.py --batch-size 32

# 使用混合精度
python benchmarks/benchmark_resnet50.py --amp

# 使用 FSDP 降低显存占用
python benchmarks/benchmark_resnet50_fsdp.py
```

### 5. 只有单卡在训练

确保使用正确的启动方式：

```bash
# 推荐：使用 torchrun
torchrun --nproc_per_node=8 your_script.py

# 或者脚本会自动使用 mp.spawn
python your_script.py
```

## ❓ 常见问题

### Q: Docker 容器中如何运行？

**A:** 需要正确配置容器参数：

```bash
docker run \
  --gpus all \              # 启用所有 GPU
  --ipc=host \             # 共享内存（重要！）
  --ulimit memlock=-1 \    # 解除内存锁定限制
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  your_image \
  bash scripts/test_8gpu.sh
```

### Q: 如何选择 DDP 还是 FSDP？

**A:** 根据模型大小选择：

- **DDP**: 适合中小型模型（< 10B 参数），通信开销小，速度快
- **FSDP**: 适合大型模型（> 10B 参数），显存占用少，可训练超大模型

详细对比见 `docs/DDP_vs_FSDP_COMPARISON.md`

### Q: 如何监控训练过程？

**A:** 提供多种监控方式：

```bash
# 1. 实时 GPU 监控
bash scripts/monitor_gpu.sh

# 2. 使用 SwanLab 记录和可视化
python benchmarks/benchmark_resnet50_maxmem.py

# 3. 查看训练日志
tail -f gpu_test_*.log
```

### Q: 测试建议的执行顺序？

**A:** 推荐按以下顺序进行：

1. **环境检测** (5 分钟)
   ```bash
   bash scripts/test_8gpu.sh
   ```

2. **通信测试** (1 分钟)
   ```bash
   python tests/test_nccl_quick.py
   ```

3. **训练测试** (2 分钟)
   ```bash
   python tests/test_ddp.py
   ```

4. **性能基准** (5 分钟)
   ```bash
   python benchmarks/benchmark_resnet50.py --iterations 200
   ```

## 📚 更多文档

- **[DDP vs FSDP 对比](docs/DDP_vs_FSDP_COMPARISON.md)** - 两种分布式策略的详细对比
- **[性能测试总结](docs/PERFORMANCE_SUMMARY.md)** - 各种配置下的性能数据
- **[SwanLab 使用指南](docs/MAXMEM_SWANLAB_GUIDE.md)** - 训练监控和可视化
- **[NCCL 诊断指南](docs/NCCL_SHM_DIAGNOSIS.md)** - NCCL 问题排查

## 🔗 相关资源

- [PyTorch 分布式训练文档](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVIDIA 深度学习性能指南](https://docs.nvidia.com/deeplearning/performance/index.html)
- [PyTorch FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## 📝 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**快速帮助:**
- 🐛 遇到问题？查看 [故障排查](#-故障排查) 和 [常见问题](#-常见问题)
- 📊 想了解性能？查看 [性能基准](#-性能基准)
- 📖 需要详细文档？查看 [docs/](docs/) 目录

**项目维护**: 定期更新以支持最新的 PyTorch 和 CUDA 版本
