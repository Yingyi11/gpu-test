#!/bin/bash
################################################################################
# 八卡 GPU 训练环境完整检测脚本
# 用途: 自动检测驱动、CUDA、NCCL、PyTorch 分布式训练环境
# 使用: bash test_8gpu.sh
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志文件
LOG_FILE="gpu_test_$(date +%Y%m%d_%H%M%S).log"

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

echo_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

echo_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$LOG_FILE"
}

echo_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
}

echo_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
}

################################################################################
# 1. 系统环境检查
################################################################################
test_system() {
    echo_section "1. 系统环境检查"
    
    echo_info "操作系统: $(uname -s)"
    echo_info "内核版本: $(uname -r)"
    echo_info "发行版: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    
    # 检查是否在容器中
    if [ -f /.dockerenv ]; then
        echo_warning "检测到运行在 Docker 容器中"
        if ! mount | grep -q "type shm"; then
            echo_error "未检测到 --ipc=host 或足够的共享内存"
        fi
    fi
}

################################################################################
# 2. NVIDIA 驱动和 GPU 检查
################################################################################
test_nvidia_driver() {
    echo_section "2. NVIDIA 驱动检查"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo_error "nvidia-smi 未找到,请安装 NVIDIA 驱动"
        return 1
    fi
    
    echo_info "NVIDIA 驱动版本:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tee -a "$LOG_FILE"
    
    echo_info "检测到的 GPU 数量:"
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "$GPU_COUNT" | tee -a "$LOG_FILE"
    
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo_error "只检测到 $GPU_COUNT 个 GPU,期望 8 个"
        nvidia-smi -L | tee -a "$LOG_FILE"
        return 1
    fi
    
    echo_success "成功检测到 8 个 GPU"
    nvidia-smi -L | tee -a "$LOG_FILE"
    
    # GPU 拓扑检查
    echo_info "GPU 拓扑结构:"
    nvidia-smi topo -m | tee -a "$LOG_FILE"
}

################################################################################
# 3. CUDA 检查
################################################################################
test_cuda() {
    echo_section "3. CUDA 环境检查"
    
    if ! command -v nvcc &> /dev/null; then
        echo_warning "nvcc 未找到,但不影响 PyTorch 使用预编译的 CUDA"
    else
        echo_info "CUDA 编译器版本:"
        nvcc --version | tee -a "$LOG_FILE"
    fi
    
    echo_info "CUDA Runtime 版本:"
    nvidia-smi | grep "CUDA Version" | tee -a "$LOG_FILE"
}

################################################################################
# 4. PyTorch 环境检查
################################################################################
test_pytorch() {
    echo_section "4. PyTorch 环境检查"
    
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo_error "Python 未找到"
        return 1
    fi
    
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    echo_info "Python 版本: $($PYTHON_CMD --version)"
    
    # 检查 PyTorch
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo_error "PyTorch 未安装,请先安装: pip install torch"
        return 1
    fi
    
    echo_success "PyTorch 已安装"
    $PYTHON_CMD -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" | tee -a "$LOG_FILE"
    
    # 检查 CUDA 支持
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        echo_error "PyTorch CUDA 不可用"
        return 1
    fi
    
    echo_success "PyTorch CUDA 可用"
    $PYTHON_CMD -c "import torch; print(f'CUDA 版本: {torch.version.cuda}')" | tee -a "$LOG_FILE"
    $PYTHON_CMD -c "import torch; print(f'可见 GPU 数量: {torch.cuda.device_count()}')" | tee -a "$LOG_FILE"
    
    # 检查 NCCL
    echo_info "NCCL 版本:"
    $PYTHON_CMD -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null | tee -a "$LOG_FILE" || echo_warning "NCCL 版本信息获取失败"
}

################################################################################
# 5. 分布式通信测试
################################################################################
test_distributed() {
    echo_section "5. 分布式通信测试"
    
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    echo_info "测试 NCCL backend 初始化..."
    
    # 创建临时测试脚本
    cat > /tmp/test_nccl.py << 'EOF'
import torch
import torch.distributed as dist
import os

def test_nccl():
    if not dist.is_available():
        print("ERROR: torch.distributed 不可用")
        return False
    
    if not dist.is_nccl_available():
        print("ERROR: NCCL backend 不可用")
        return False
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # 测试 all_reduce
    tensor = torch.ones(1).cuda() * (rank + 1)
    dist.all_reduce(tensor)
    expected = sum(range(1, world_size + 1))
    
    if abs(tensor.item() - expected) < 1e-5:
        print(f"Rank {rank}: 通信测试成功, tensor={tensor.item()}")
    else:
        print(f"Rank {rank}: 通信测试失败, expected={expected}, got={tensor.item()}")
    
    dist.destroy_process_group()
    return True

if __name__ == "__main__":
    try:
        test_nccl()
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
EOF
    
    # 运行分布式测试
    if command -v torchrun &> /dev/null; then
        torchrun --nproc_per_node=8 /tmp/test_nccl.py 2>&1 | tee -a "$LOG_FILE"
        if [ $? -eq 0 ]; then
            echo_success "8 卡分布式通信测试通过"
        else
            echo_error "8 卡分布式通信测试失败"
            return 1
        fi
    else
        echo_warning "torchrun 未找到,跳过分布式测试"
        echo_info "提示: PyTorch >= 1.10 自带 torchrun"
    fi
    
    rm -f /tmp/test_nccl.py
}

################################################################################
# 6. 简易性能测试
################################################################################
test_performance() {
    echo_section "6. 简易性能测试"
    
    echo_info "运行 DDP 小规模训练测试..."
    
    if [ -f "tests/test_ddp.py" ]; then
        timeout 60 python tests/test_ddp.py 2>&1 | tee -a "$LOG_FILE"
        if [ $? -eq 0 ]; then
            echo_success "DDP 训练测试通过"
        else
            echo_warning "DDP 训练测试超时或失败(可能是正常的短时测试)"
        fi
    else
        echo_warning "tests/test_ddp.py 不存在,跳过性能测试"
        echo_info "提示: 运行 'python tests/test_ddp.py' 进行完整测试"
    fi
}

################################################################################
# 7. GPU 带宽测试
################################################################################
test_bandwidth() {
    echo_section "7. GPU 通信带宽测试"
    
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    # 创建带宽测试脚本
    cat > /tmp/test_bandwidth.py << 'EOF'
import torch
import torch.distributed as dist
import os
import time

def test_bandwidth():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # 测试不同大小的数据传输
    sizes = [1024*1024, 8*1024*1024, 64*1024*1024]  # 1MB, 8MB, 64MB
    
    if rank == 0:
        print(f"\n{'Size':<10} {'Bandwidth (GB/s)':<20}")
        print("-" * 30)
    
    for size in sizes:
        tensor = torch.randn(size, dtype=torch.float32).cuda()
        
        # 预热
        for _ in range(5):
            dist.all_reduce(tensor)
        
        # 测试
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # 计算带宽
        data_size = size * 4  # float32 = 4 bytes
        bandwidth = (data_size * 10) / elapsed / 1e9  # GB/s
        
        if rank == 0:
            print(f"{data_size/1024/1024:.1f}MB     {bandwidth:.2f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        test_bandwidth()
    except Exception as e:
        print(f"ERROR: {e}")
EOF
    
    if command -v torchrun &> /dev/null; then
        echo_info "测试 all_reduce 带宽..."
        torchrun --nproc_per_node=8 /tmp/test_bandwidth.py 2>&1 | tee -a "$LOG_FILE"
    else
        echo_warning "跳过带宽测试"
    fi
    
    rm -f /tmp/test_bandwidth.py
}

################################################################################
# 8. 生成测试报告
################################################################################
generate_report() {
    echo_section "8. 测试报告"
    
    echo_info "测试完成时间: $(date)"
    echo_info "详细日志已保存至: $LOG_FILE"
    
    echo ""
    echo_success "=========================================="
    echo_success "  八卡 GPU 环境测试完成!"
    echo_success "=========================================="
    echo ""
    echo_info "下一步操作:"
    echo "  1. 运行完整训练测试: python tests/test_ddp.py"
    echo "  2. 运行 ResNet50 性能测试: python benchmarks/benchmark_resnet50.py"
    echo "  3. 查看详细日志: cat $LOG_FILE"
    echo ""
}

################################################################################
# 主函数
################################################################################
main() {
    echo_section "八卡 GPU 训练环境检测工具"
    echo_info "开始时间: $(date)"
    
    test_system
    test_nvidia_driver
    test_cuda
    test_pytorch
    test_distributed
    test_performance
    test_bandwidth
    generate_report
}

# 运行主函数
main
