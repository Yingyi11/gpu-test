#!/usr/bin/env python3
"""
ResNet50 八卡性能基准测试
测试实际模型的训练吞吐量和 GPU 利用率

使用方式:
    python benchmark_resnet50.py                        # 默认配置
    python benchmark_resnet50.py --batch-size 128       # 自定义 batch size
    torchrun --nproc_per_node=8 benchmark_resnet50.py   # 使用 torchrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import time
import argparse
import os
from datetime import datetime


class SyntheticDataset(Dataset):
    """合成数据集,用于性能测试"""
    def __init__(self, size=10000, image_size=224):
        self.size = size
        self.image_size = image_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机图像和标签
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, 1000, (1,)).item()
        return image, label


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 方式
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化分布式进程组
        dist.init_process_group(backend='nccl')
    else:
        # 单 GPU 模式,不使用分布式
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_model(model_name='resnet50', pretrained=False):
    """创建模型"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model


def benchmark(args):
    """性能基准测试主函数"""
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # PyTorch 性能优化设置
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 (Ampere+)
    torch.backends.cudnn.allow_tf32 = True
    
    # 打印配置信息
    if rank == 0:
        print("\n" + "="*80)
        print(f"ResNet50 性能基准测试")
        print("="*80)
        print(f"配置信息:")
        print(f"  - GPU 数量: {world_size}")
        print(f"  - 模型: {args.model}")
        print(f"  - 批次大小(每 GPU): {args.batch_size}")
        print(f"  - 全局批次大小: {args.batch_size * world_size}")
        print(f"  - 迭代次数: {args.iterations}")
        print(f"  - 预热迭代: {args.warmup}")
        print(f"  - 数据加载线程: {args.workers}")
        print(f"  - 混合精度: {args.fp16 or args.amp}")
        print(f"  - 优化器: {args.optimizer}")
        print(f"  - 学习率: {args.lr}")
        print(f"\n优化设置:")
        print(f"  - cuDNN benchmark: True")
        print(f"  - TF32: True")
        print(f"  - DDP gradient_as_bucket_view: {world_size > 1}")
        print(f"  - DataLoader persistent_workers: True")
        print(f"  - DataLoader prefetch_factor: 4")
        print("="*80 + "\n")
    
    # 创建模型
    model = create_model(args.model, pretrained=False).to(device)
    
    # 使用 DDP 包装模型,启用性能优化选项
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank],
            broadcast_buffers=False,  # 不广播 buffer,减少通信
            gradient_as_bucket_view=True,  # 使用 bucket view 减少内存拷贝
            find_unused_parameters=False  # 加速,假设所有参数都参与梯度计算
        )
    
    # 创建优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 创建数据集和数据加载器
    # 对于分布式训练,需要确保有足够的数据供所有 GPU 使用
    dataset_size = args.iterations * args.batch_size * world_size
    dataset = SyntheticDataset(size=dataset_size)
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=True if args.workers > 0 else False,  # 保持 workers 存活
            prefetch_factor=4 if args.workers > 0 else None  # 预取更多批次
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=True if args.workers > 0 else False,
            prefetch_factor=4 if args.workers > 0 else None
        )
    
    # 混合精度训练
    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # 预热
    if rank == 0:
        print("预热阶段...")
    
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        if i >= args.warmup:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 同步
    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize()
    
    # 性能测试
    if rank == 0:
        print(f"\n开始性能测试 ({args.iterations} 次迭代)...\n")
    
    times = []
    throughputs = []
    
    start_time = time.time()
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= args.iterations:
            break
        
        iter_start = time.time()
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        
        # 计算吞吐量
        throughput = (args.batch_size * world_size) / iter_time
        
        times.append(iter_time)
        throughputs.append(throughput)
        
        if rank == 0 and (i + 1) % 10 == 0:
            avg_throughput = sum(throughputs[-10:]) / len(throughputs[-10:])
            print(f"[Iter {i+1:3d}/{args.iterations}] "
                  f"Time: {iter_time*1000:.1f}ms | "
                  f"Throughput: {throughput:.1f} images/s | "
                  f"Avg(last 10): {avg_throughput:.1f} images/s")
    
    total_time = time.time() - start_time
    
    # 同步
    if world_size > 1:
        dist.barrier()
    
    # 打印统计信息
    if rank == 0:
        avg_time = sum(times) / len(times)
        avg_throughput = sum(throughputs) / len(throughputs)
        
        print("\n" + "="*80)
        print("性能统计:")
        print("="*80)
        print(f"总时间: {total_time:.2f}s")
        print(f"平均迭代时间: {avg_time*1000:.2f}ms")
        print(f"平均吞吐量: {avg_throughput:.1f} images/s")
        print(f"总处理图像: {args.iterations * args.batch_size * world_size}")
        print(f"每 GPU 吞吐量: {avg_throughput / world_size:.1f} images/s")
        
        # 计算统计指标
        import statistics
        if len(times) > 1:
            median_time = statistics.median(times)
            p95_time = sorted(times)[int(len(times) * 0.95)]
            p99_time = sorted(times)[int(len(times) * 0.99)]
            
            print(f"\n迭代时间分布:")
            print(f"  - 中位数: {median_time*1000:.2f}ms")
            print(f"  - P95: {p95_time*1000:.2f}ms")
            print(f"  - P99: {p99_time*1000:.2f}ms")
        
        # GPU 显存使用
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"\nGPU 显存使用:")
        print(f"  - 已分配: {memory_allocated:.2f} GB")
        print(f"  - 已预留: {memory_reserved:.2f} GB")
        
        print("="*80 + "\n")
    
    # 清理
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='ResNet50 性能基准测试')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='模型名称')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64,
                        help='每 GPU 的批次大小')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试迭代次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热迭代次数')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数 (建议 8-16 用于多 GPU)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学习率')
    
    # 混合精度
    parser.add_argument('--fp16', action='store_true',
                        help='使用 FP16 训练')
    parser.add_argument('--amp', action='store_true',
                        help='使用 PyTorch AMP 混合精度')
    
    args = parser.parse_args()
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用!")
        return
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"检测到 {torch.cuda.device_count()} 个 GPU\n")
    
    # 运行基准测试
    try:
        benchmark(args)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()


if __name__ == "__main__":
    # 设置环境变量以优化 NCCL 性能
    os.environ['NCCL_DEBUG'] = 'WARN'  # 减少 NCCL 日志
    
    # 共享内存问题的解决方案 - 系统不支持时必须禁用
    os.environ['NCCL_SHM_DISABLE'] = '1'  # 禁用共享内存,使用网络通信
    
    # NCCL 性能优化
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用 InfiniBand (本地训练不需要)
    os.environ['NCCL_P2P_DISABLE'] = '0'  # 启用 P2P (GPU 间直接通信)
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用 loopback 接口
    
    # PyTorch 优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 启用异步 CUDA 操作
    
    start = time.time()
    print(f"\n{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    main()
    
    print(f"\n{'='*80}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {time.time() - start:.2f}s")
    print(f"{'='*80}\n")
