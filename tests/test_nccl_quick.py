#!/usr/bin/env python3
"""
快速 NCCL 通信测试脚本
不涉及训练,只测试 GPU 间的通信能力

使用方式:
    python test_nccl_quick.py
    torchrun --nproc_per_node=8 test_nccl_quick.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from datetime import datetime


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def test_all_reduce(rank, world_size, size_mb=1):
    """测试 all_reduce 操作"""
    device = torch.device(f'cuda:{rank}')
    
    # 创建测试数据
    num_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    tensor = torch.ones(num_elements, dtype=torch.float32).to(device) * (rank + 1)
    
    # 预热
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    
    # 测试
    num_iterations = 20
    start = time.time()
    
    for _ in range(num_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = num_elements * 4  # bytes
    bandwidth = (data_size * num_iterations) / elapsed / 1e9  # GB/s
    
    # 验证正确性
    expected_sum = sum(range(1, world_size + 1)) * num_elements
    is_correct = abs(tensor.sum().item() - expected_sum) < 1e-3
    
    return bandwidth, is_correct


def test_broadcast(rank, world_size, size_mb=1):
    """测试 broadcast 操作"""
    device = torch.device(f'cuda:{rank}')
    
    num_elements = (size_mb * 1024 * 1024) // 4
    
    if rank == 0:
        tensor = torch.ones(num_elements, dtype=torch.float32).to(device) * 42
    else:
        tensor = torch.zeros(num_elements, dtype=torch.float32).to(device)
    
    # 预热
    for _ in range(5):
        dist.broadcast(tensor, src=0)
    
    torch.cuda.synchronize()
    
    # 测试
    num_iterations = 20
    start = time.time()
    
    for _ in range(num_iterations):
        dist.broadcast(tensor, src=0)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = num_elements * 4
    bandwidth = (data_size * num_iterations) / elapsed / 1e9
    
    # 验证正确性
    is_correct = abs(tensor.sum().item() - 42 * num_elements) < 1e-3
    
    return bandwidth, is_correct


def test_all_gather(rank, world_size, size_mb=1):
    """测试 all_gather 操作"""
    device = torch.device(f'cuda:{rank}')
    
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.ones(num_elements, dtype=torch.float32).to(device) * (rank + 1)
    tensor_list = [torch.zeros(num_elements, dtype=torch.float32).to(device) 
                   for _ in range(world_size)]
    
    # 预热
    for _ in range(5):
        dist.all_gather(tensor_list, tensor)
    
    torch.cuda.synchronize()
    
    # 测试
    num_iterations = 20
    start = time.time()
    
    for _ in range(num_iterations):
        dist.all_gather(tensor_list, tensor)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = num_elements * 4
    bandwidth = (data_size * num_iterations * world_size) / elapsed / 1e9
    
    # 验证正确性
    is_correct = all(
        abs(tensor_list[i].sum().item() - (i + 1) * num_elements) < 1e-3
        for i in range(world_size)
    )
    
    return bandwidth, is_correct


def test_reduce_scatter(rank, world_size, size_mb=1):
    """测试 reduce_scatter 操作"""
    device = torch.device(f'cuda:{rank}')
    
    num_elements = (size_mb * 1024 * 1024) // 4
    chunk_size = num_elements // world_size
    
    input_tensor = torch.ones(num_elements, dtype=torch.float32).to(device) * (rank + 1)
    output_tensor = torch.zeros(chunk_size, dtype=torch.float32).to(device)
    
    input_list = list(input_tensor.chunk(world_size))
    
    # 预热
    for _ in range(5):
        dist.reduce_scatter(output_tensor, input_list)
    
    torch.cuda.synchronize()
    
    # 测试
    num_iterations = 20
    start = time.time()
    
    for _ in range(num_iterations):
        dist.reduce_scatter(output_tensor, input_list)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = num_elements * 4
    bandwidth = (data_size * num_iterations) / elapsed / 1e9
    
    return bandwidth, True


def run_tests(rank, world_size):
    """运行所有测试"""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    setup_distributed(rank, world_size)
    
    # 打印头部
    if rank == 0:
        print("\n" + "="*80)
        print("NCCL 通信性能测试")
        print("="*80)
        print(f"GPU 数量: {world_size}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 版本: {torch.version.cuda}")
        try:
            nccl_version = torch.cuda.nccl.version()
            print(f"NCCL 版本: {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
        except:
            pass
        print("="*80 + "\n")
    
    # 测试不同数据大小
    test_sizes = [1, 4, 16, 64]  # MB
    
    results = {
        'all_reduce': [],
        'broadcast': [],
        'all_gather': [],
        'reduce_scatter': []
    }
    
    for size_mb in test_sizes:
        # all_reduce
        bandwidth, is_correct = test_all_reduce(rank, world_size, size_mb)
        results['all_reduce'].append((size_mb, bandwidth, is_correct))
        
        # broadcast
        bandwidth, is_correct = test_broadcast(rank, world_size, size_mb)
        results['broadcast'].append((size_mb, bandwidth, is_correct))
        
        # all_gather
        bandwidth, is_correct = test_all_gather(rank, world_size, size_mb)
        results['all_gather'].append((size_mb, bandwidth, is_correct))
        
        # reduce_scatter
        bandwidth, is_correct = test_reduce_scatter(rank, world_size, size_mb)
        results['reduce_scatter'].append((size_mb, bandwidth, is_correct))
    
    # 同步并打印结果
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*80)
        print("测试结果")
        print("="*80 + "\n")
        
        for op_name, op_results in results.items():
            print(f"{op_name.upper()}:")
            print(f"  {'Size (MB)':<12} {'Bandwidth (GB/s)':<20} {'Status':<10}")
            print(f"  {'-'*12} {'-'*20} {'-'*10}")
            
            for size_mb, bandwidth, is_correct in op_results:
                status = "✓" if is_correct else "✗"
                print(f"  {size_mb:<12} {bandwidth:<20.2f} {status:<10}")
            
            print()
        
        print("="*80)
        print("✓ 所有测试完成!")
        print("="*80 + "\n")
    
    cleanup_distributed()


def test_gpu_peer_access(rank, world_size):
    """测试 GPU 点对点访问"""
    torch.cuda.set_device(rank)
    
    setup_distributed(rank, world_size)
    
    if rank == 0:
        print("\n" + "="*80)
        print("GPU P2P 访问测试")
        print("="*80 + "\n")
        
        for i in range(world_size):
            for j in range(world_size):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    status = "✓" if can_access else "✗"
                    print(f"GPU {i} -> GPU {j}: {status}")
        
        print("\n" + "="*80 + "\n")
    
    dist.barrier()
    cleanup_distributed()


def main():
    """主函数"""
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用!")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"\n检测到 {gpu_count} 个 GPU")
    
    if gpu_count < 8:
        print(f"警告: 只检测到 {gpu_count} 个 GPU,将使用所有可用 GPU")
    
    world_size = min(gpu_count, 8)
    
    # 检查是否在分布式环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 启动
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        run_tests(rank, world_size)
    else:
        # mp.spawn 启动
        print(f"使用 {world_size} 个 GPU 进行测试\n")
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        
        # P2P 测试
        mp.spawn(
            test_gpu_peer_access,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        
        # 通信性能测试
        mp.spawn(
            run_tests,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    # 设置环境变量
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    start_time = time.time()
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time:.2f} 秒\n")
