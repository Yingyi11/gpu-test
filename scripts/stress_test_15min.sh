#!/bin/bash
#
# 15分钟 GPU 压力测试脚本
# 测试 8 卡分布式训练的稳定性和持续性能
#

echo "================================================================================"
echo "                     15分钟 GPU 分布式训练压力测试"
echo "                          显存最大化 + SwanLab监控版本"
echo "================================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "测试配置:"
echo "  - 持续时间: 15 分钟"
echo "  - GPU 数量: 8"
echo "  - 模型: ResNet50"
echo "  - Batch Size: 256/GPU (全局 2048) - 最大化显存使用"
echo "  - 混合精度: FP32"
echo "  - Workers: 12"
echo "  - 目标显存: 20 GB/GPU"
echo "  - SwanLab监控: 启用"
echo ""
echo "测试目的:"
echo "  ✓ 验证长时间运行稳定性"
echo "  ✓ 检测热管理和温度控制"
echo "  ✓ 监控性能是否随时间衰减"
echo "  ✓ 测试 NCCL 通信可靠性"
echo "  ✓ 测试显存满载下的稳定性"
echo "  ✓ 实时监控并上传GPU指标到SwanLab"
echo "================================================================================"
echo ""

# 计算需要的迭代次数
# batch size 256 比 128 慢约2倍
# 假设每次迭代约 0.40 秒
# 15 分钟 = 900 秒
# 900 / 0.40 ≈ 2250 次迭代
# 设置 2300 次迭代以确保至少15分钟

ITERATIONS=2300
BATCH_SIZE=256
WORKERS=12
WARMUP=20
TARGET_MEMORY=20

echo "计算的迭代次数: $ITERATIONS (预计运行时间: ~15分钟)"
echo "预热迭代: $WARMUP"
echo "目标显存使用: $TARGET_MEMORY GB/GPU"
echo ""
echo "开始压力测试 (显存最大化版本)..."
echo ""

# 运行测试 - 使用增强版脚本
uv run torchrun \
    --nproc_per_node=8 \
    --master_port=29510 \
    benchmarks/benchmark_resnet50_maxmem.py \
    --batch-size $BATCH_SIZE \
    --iterations $ITERATIONS \
    --warmup $WARMUP \
    --workers $WORKERS \
    --target-memory $TARGET_MEMORY \
    --enable-swanlab \
    --swanlab-project "gpu-stress-test" \
    --monitor-interval 5

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "                            压力测试完成"
echo "================================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "状态: ✅ 成功 - 系统通过 15 分钟显存满载压力测试"
    echo ""
    echo "监控数据已上传到 SwanLab"
    echo "访问 https://swanlab.cn 查看详细监控数据"
else
    echo "状态: ❌ 失败 - 退出码 $EXIT_CODE"
fi

echo "================================================================================"
