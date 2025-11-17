#!/usr/bin/env python3
"""
ResNet50 八卡性能基准测试 - 显存最大化版本
特点:
1. 使用更大的batch size和模型来吃满显存
2. 集成SwanLab进行GPU监控数据上传
3. 实时监控GPU温度、利用率、显存、功率等指标

使用方式:
    torchrun --nproc_per_node=8 benchmark_resnet50_maxmem.py --batch-size 256
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
import threading
from datetime import datetime

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("警告: SwanLab 未安装,监控功能将被禁用")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("警告: pynvml 未安装,GPU监控功能将被禁用")


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


class GPUMonitor:
    """GPU 监控器 - 收集并上传 GPU 指标到 SwanLab"""
    def __init__(self, interval=5, enabled=True, rank=0):
        self.interval = interval
        self.enabled = enabled and PYNVML_AVAILABLE
        self.rank = rank
        self.running = False
        self.thread = None
        
        if self.enabled and rank == 0:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
    
    def start(self):
        """启动监控线程"""
        if not self.enabled or self.rank != 0:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"GPU 监控已启动 (采样间隔: {self.interval}秒)")
    
    def stop(self):
        """停止监控线程"""
        if not self.enabled or self.rank != 0:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
        print("GPU 监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 收集所有GPU的指标
                metrics = {}
                total_memory_used = 0
                total_memory_total = 0
                total_util = 0
                total_power = 0
                max_temp = 0
                
                for i, handle in enumerate(self.handles):
                    # 获取GPU指标
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        
                        # 单GPU指标
                        metrics[f'gpu_{i}/utilization'] = util.gpu
                        metrics[f'gpu_{i}/memory_used_mb'] = memory.used / 1024**2
                        metrics[f'gpu_{i}/memory_util'] = (memory.used / memory.total) * 100
                        metrics[f'gpu_{i}/temperature'] = temp
                        metrics[f'gpu_{i}/power_w'] = power
                        
                        # 累加用于计算平均值
                        total_memory_used += memory.used
                        total_memory_total += memory.total
                        total_util += util.gpu
                        total_power += power
                        max_temp = max(max_temp, temp)
                    except Exception as e:
                        print(f"警告: 无法获取 GPU {i} 的指标: {e}")
                        continue
                
                # 平均指标
                if self.device_count > 0:
                    metrics['gpu_avg/utilization'] = total_util / self.device_count
                    metrics['gpu_avg/memory_used_gb'] = total_memory_used / (1024**3)
                    metrics['gpu_avg/memory_util'] = (total_memory_used / total_memory_total) * 100
                    metrics['gpu_avg/temperature_max'] = max_temp
                    metrics['gpu_avg/power_total_w'] = total_power
                
                # 上传到 SwanLab
                if SWANLAB_AVAILABLE:
                    swanlab.log(metrics)
                
            except Exception as e:
                print(f"监控错误: {e}")
            
            time.sleep(self.interval)
    
    def __del__(self):
        """清理"""
        if self.enabled and self.rank == 0:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


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


def allocate_extra_memory(device, target_gb=20):
    """
    预分配显存以达到目标显存使用量
    返回分配的tensor列表以保持引用
    """
    buffers = []
    try:
        # 获取当前显存使用情况
        current_gb = torch.cuda.memory_allocated(device) / 1024**3
        target_bytes = int((target_gb - current_gb) * 1024**3)
        
        if target_bytes > 0:
            print(f"  预分配显存: {target_bytes / 1024**3:.2f} GB")
            # 分配大块连续内存
            buffer = torch.randn(target_bytes // 4, dtype=torch.float32, device=device)
            buffers.append(buffer)
            
            # 强制同步确保内存分配完成
            torch.cuda.synchronize()
    except Exception as e:
        print(f"  显存预分配警告: {e}")
    
    return buffers


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
    
    # 初始化 SwanLab (仅 rank 0)
    if rank == 0 and SWANLAB_AVAILABLE and args.enable_swanlab:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"gpu_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"8-GPU ResNet50 Full Memory Stress Test",
            config={
                "model": args.model,
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "iterations": args.iterations,
                "workers": args.workers,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "target_memory_gb": args.target_memory,
            }
        )
    
    # 打印配置信息
    if rank == 0:
        print("\n" + "="*80)
        print(f"ResNet50 性能基准测试 - 显存最大化版本")
        print("="*80)
        print(f"配置信息:")
        print(f"  - GPU 数量: {world_size}")
        print(f"  - 模型: {args.model}")
        print(f"  - 批次大小(每 GPU): {args.batch_size}")
        print(f"  - 全局批次大小: {args.batch_size * world_size}")
        print(f"  - 迭代次数: {args.iterations}")
        print(f"  - 预热迭代: {args.warmup}")
        print(f"  - 数据加载线程: {args.workers}")
        print(f"  - 混合精度: {args.amp}")
        print(f"  - 优化器: {args.optimizer}")
        print(f"  - 学习率: {args.lr}")
        print(f"  - 目标显存使用: {args.target_memory} GB/GPU")
        print(f"  - SwanLab监控: {args.enable_swanlab and SWANLAB_AVAILABLE}")
        print(f"\n优化设置:")
        print(f"  - cuDNN benchmark: True")
        print(f"  - TF32: True")
        print(f"  - DDP gradient_as_bucket_view: {world_size > 1}")
        print(f"  - DataLoader persistent_workers: True")
        print(f"  - DataLoader prefetch_factor: 4")
        print("="*80 + "\n")
    
    # 启动 GPU 监控 (仅 rank 0)
    gpu_monitor = GPUMonitor(
        interval=args.monitor_interval,
        enabled=args.enable_swanlab and SWANLAB_AVAILABLE,
        rank=rank
    )
    gpu_monitor.start()
    
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
    
    # 预分配显存以达到目标使用量
    if rank == 0:
        print("准备显存分配...")
    
    memory_buffers = []
    if args.target_memory > 0:
        memory_buffers = allocate_extra_memory(device, args.target_memory)
    
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
    
    # 显存使用情况
    if rank == 0:
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"\n预热后显存使用:")
        print(f"  - 已分配: {memory_allocated:.2f} GB")
        print(f"  - 已预留: {memory_reserved:.2f} GB\n")
    
    # 性能测试
    if rank == 0:
        print(f"开始性能测试 ({args.iterations} 次迭代)...\n")
    
    times = []
    throughputs = []
    losses = []
    
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
        losses.append(loss.item())
        
        # 上传训练指标到 SwanLab
        if rank == 0 and SWANLAB_AVAILABLE and args.enable_swanlab:
            swanlab.log({
                'train/loss': loss.item(),
                'train/throughput': throughput,
                'train/iter_time_ms': iter_time * 1000,
                'train/step': i + 1,
            })
        
        if rank == 0 and (i + 1) % 10 == 0:
            avg_throughput = sum(throughputs[-10:]) / len(throughputs[-10:])
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"[Iter {i+1:4d}/{args.iterations}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {iter_time*1000:.1f}ms | "
                  f"Throughput: {throughput:.1f} imgs/s | "
                  f"Avg(last 10): {avg_throughput:.1f} imgs/s")
    
    total_time = time.time() - start_time
    
    # 同步
    if world_size > 1:
        dist.barrier()
    
    # 停止监控
    gpu_monitor.stop()
    
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
        
        # 上传最终统计到 SwanLab
        if SWANLAB_AVAILABLE and args.enable_swanlab:
            swanlab.log({
                'summary/avg_throughput': avg_throughput,
                'summary/avg_iter_time_ms': avg_time * 1000,
                'summary/total_time_s': total_time,
                'summary/max_memory_allocated_gb': memory_allocated,
                'summary/max_memory_reserved_gb': memory_reserved,
            })
        
        print("="*80 + "\n")
    
    # 关闭 SwanLab
    if rank == 0 and SWANLAB_AVAILABLE and args.enable_swanlab:
        swanlab.finish()
    
    # 清理
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='ResNet50 性能基准测试 - 显存最大化版本')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='模型名称')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=256,
                        help='每 GPU 的批次大小 (默认256以最大化显存使用)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试迭代次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热迭代次数')
    parser.add_argument('--workers', type=int, default=12,
                        help='数据加载线程数 (建议 8-16 用于多 GPU)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学习率')
    
    # 混合精度
    parser.add_argument('--amp', action='store_true',
                        help='使用 PyTorch AMP 混合精度')
    
    # 显存管理
    parser.add_argument('--target-memory', type=float, default=0,
                        help='目标显存使用量 (GB), 0表示不预分配')
    
    # SwanLab 监控
    parser.add_argument('--enable-swanlab', action='store_true',
                        help='启用 SwanLab 监控')
    parser.add_argument('--swanlab-project', type=str, default='gpu-benchmark',
                        help='SwanLab 项目名称')
    parser.add_argument('--monitor-interval', type=int, default=5,
                        help='GPU 监控采样间隔 (秒)')
    
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
