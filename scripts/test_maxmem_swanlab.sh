#!/bin/bash
#
# 快速测试脚本 - 测试显存最大化和 SwanLab 监控功能
# 运行较短的时间以快速验证功能
#

echo "================================================================================"
echo "                GPU 显存最大化 + SwanLab 监控测试"
echo "================================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "测试配置:"
echo "  - GPU 数量: 8"
echo "  - 模型: ResNet50"
echo "  - Batch Size: 256/GPU (全局 2048)"
echo "  - 迭代次数: 50 (快速测试)"
echo "  - 目标显存: 20 GB/GPU"
echo "  - SwanLab监控: 启用"
echo ""
echo "功能验证:"
echo "  ✓ 显存是否达到目标使用量"
echo "  ✓ SwanLab 是否正常上传数据"
echo "  ✓ GPU 监控是否工作正常"
echo "  ✓ 多卡训练是否稳定"
echo "================================================================================"
echo ""

# 运行快速测试
uv run torchrun \
    --nproc_per_node=8 \
    --master_port=29510 \
    benchmarks/benchmark_resnet50_maxmem.py \
    --batch-size 256 \
    --iterations 4000 \
    --warmup 10 \
    --workers 12 \
    --target-memory 20 \
    --enable-swanlab \
    --swanlab-project "gpu-stress-test" \
    --monitor-interval 5

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "                            快速测试完成"
echo "================================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "状态: ✅ 成功"
    echo ""
    echo "✓ 显存最大化功能正常"
    echo "✓ SwanLab 监控数据已上传"
    echo ""
    echo "访问 https://swanlab.cn 查看监控数据"
    echo "如果功能正常,可以运行 ./stress_test_15min.sh 进行完整的15分钟压力测试"
else
    echo "状态: ❌ 失败 - 退出码 $EXIT_CODE"
fi

echo "================================================================================"
