#!/usr/bin/env python3
"""
八卡 DDP (DistributedDataParallel) 训练测试脚本
测试 PyTorch 分布式训练是否正常工作

使用方式:
    python test_ddp.py                    # 使用 mp.spawn 自动启动 8 进程
    torchrun --nproc_per_node=8 test_ddp.py  # 使用 torchrun 启动
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime


class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=1000, hidden_size=2000, output_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_ddp(rank, world_size, num_epochs=5, batch_size=64):
    """
    DDP 训练函数
    
    Args:
        rank: 当前进程的 rank
        world_size: 总进程数(GPU 数)
        num_epochs: 训练轮数
        batch_size: 批次大小
    """
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 初始化分布式
    setup_distributed(rank, world_size)
    
    # 打印进程信息
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"开始 DDP 训练测试")
        print(f"{'='*60}")
        print(f"总 GPU 数量: {world_size}")
        print(f"批次大小: {batch_size}")
        print(f"训练轮数: {num_epochs}")
        print(f"{'='*60}\n")
    
    # 创建模型并移到对应 GPU
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 20
        
        for batch_idx in range(num_batches):
            # 生成随机数据
            inputs = torch.randn(batch_size, 1000).to(device)
            targets = torch.randn(batch_size, 1000).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        # 只在 rank 0 打印日志
        if rank == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Loss: {avg_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Throughput: {num_batches * batch_size * world_size / epoch_time:.0f} samples/s")
    
    # 测试 all_reduce 操作
    if rank == 0:
        print(f"\n{'='*60}")
        print("测试 all_reduce 操作...")
        print(f"{'='*60}")
    
    test_tensor = torch.ones(1).to(device) * (rank + 1)
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    
    print(f"[Rank {rank}] all_reduce 结果: {test_tensor.item():.0f} "
          f"(期望值: {expected_sum})")
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("✓ DDP 训练测试完成!")
        print(f"{'='*60}\n")
    
    # 清理
    cleanup_distributed()


def test_nccl_communication(rank, world_size):
    """
    测试 NCCL 通信的基本功能
    """
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    setup_distributed(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("NCCL 通信基础测试")
        print(f"{'='*60}\n")
    
    # 测试 1: all_reduce
    tensor = torch.ones(100, 100).to(device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size)) * 10000
    success = abs(tensor.sum().item() - expected) < 1e-3
    print(f"[Rank {rank}] all_reduce: {'✓' if success else '✗'}")
    
    # 测试 2: broadcast
    if rank == 0:
        tensor = torch.ones(100, 100).to(device) * 42
    else:
        tensor = torch.zeros(100, 100).to(device)
    
    dist.broadcast(tensor, src=0)
    success = abs(tensor.sum().item() - 420000) < 1e-3
    print(f"[Rank {rank}] broadcast: {'✓' if success else '✗'}")
    
    # 测试 3: all_gather
    tensor_list = [torch.zeros(100, 100).to(device) for _ in range(world_size)]
    tensor = torch.ones(100, 100).to(device) * rank
    dist.all_gather(tensor_list, tensor)
    
    success = all(
        abs(tensor_list[i].sum().item() - i * 10000) < 1e-3
        for i in range(world_size)
    )
    print(f"[Rank {rank}] all_gather: {'✓' if success else '✗'}")
    
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("✓ NCCL 通信测试完成!")
        print(f"{'='*60}\n")
    
    cleanup_distributed()


def main():
    """主函数"""
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用!")
        return
    
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个 GPU")
    
    if gpu_count < 8:
        print(f"警告: 只检测到 {gpu_count} 个 GPU,将使用所有可用 GPU 进行测试")
    
    world_size = min(gpu_count, 8)
    
    # 检查是否已经在分布式环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 使用 torchrun 启动
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"使用 torchrun 启动 (Rank {rank}/{world_size})")
        
        # 直接运行训练
        torch.cuda.set_device(local_rank)
        train_ddp(rank, world_size)
    else:
        # 使用 mp.spawn 启动
        print(f"使用 mp.spawn 启动 {world_size} 个进程\n")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 启动多进程训练
        print("=" * 60)
        print("第一阶段: NCCL 通信测试")
        print("=" * 60)
        mp.spawn(
            test_nccl_communication,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        
        print("\n" + "=" * 60)
        print("第二阶段: DDP 训练测试")
        print("=" * 60)
        mp.spawn(
            train_ddp,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    # 设置环境变量以避免一些常见问题
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '0'  # 启用 InfiniBand(如果有)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0,en,em'  # 网络接口
    
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
