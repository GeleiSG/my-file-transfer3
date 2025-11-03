import torch
import time
import sys

def occupy_gpu(gpu_id):
    """
    一个工作函数，用于持续占用单个 GPU。

    它会分配 GPU 显存的约 90%，
    并在一个无限循环中执行矩阵乘法。
    """
    
    # 检查 GPU ID 是否有效
    if not (0 <= gpu_id < torch.cuda.device_count()):
        print(f"❌ 错误: GPU ID {gpu_id} 无效。")
        print(f"   请提供一个 0 到 {torch.cuda.device_count() - 1} 之间的 ID。")
        sys.exit(1)
        
    device = f'cuda:{gpu_id}'
    
    # 设置当前设备
    torch.cuda.set_device(device)
    
    print(f"✅ [GPU {gpu_id}] 开始占用... {torch.cuda.get_device_name(gpu_id)}")

    # 初始化两个大张量以占用内存和用于计算
    a = None
    b = None
    
    try:
        # 1. 分配显存
        # 获取总显存，并计算 45% 用于每个张量 (总共 90%)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        target_bytes_per_tensor = int(total_mem * 0.45)
        
        # float32 = 4 bytes
        num_elements = target_bytes_per_tensor // 4
        # 计算一个大致的方阵维度
        size = int(num_elements**0.5)

        print(f"   [GPU {gpu_id}] 总显存: {total_mem / 1024**3:.2f} GB")
        print(f"   [GPU {gpu_id}] 正在分配两个 {size}x{size} 的张量 (每个约 {target_bytes_per_tensor / 1024**3:.2f} GB)...")

        # 分配张量
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        print(f"   [GPU {gpu_id}] 显存分配完成。开始计算循环...")

        # 2. 持续计算
        while True:
            # 执行一个高强度操作
            a = torch.add(a, 0.001) # 做一点小计算
            b = torch.add(b, 0.001)
            c = torch.matmul(a, b) # 核心计算

    except KeyboardInterrupt:
        # 捕获 Ctrl+C
        print(f"\n🛑 [GPU {gpu_id}] 收到停止信号。正在释放...")
        
    except RuntimeError as e:
        # 捕获可能的 OOM (Out of Memory) 错误
        if "out of memory" in str(e):
            print(f"\n❌ [GPU {gpu_id}] 显存不足 (OOM)！尝试分配的张量太大。")
            print("   [GPU {gpu_id}] 请尝试减小 '0.45' (45%) 这个比例。")
        else:
            print(f"\n❌ [GPU {gpu_id}] 发生运行时错误: {e}")
            
    finally:
        # 无论如何，都尝试清理资源
        if a is not None:
            del a
        if b is not None:
            del b
        
        torch.cuda.empty_cache()
        print(f"📉 [GPU {gpu_id}] 显存已释放。")

def main():
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("❌ 错误: CUDA 不可用。请检查您的 PyTorch 和 CUDA 驱动。")
        sys.exit(1)

    # 从命令行参数获取 GPU ID
    if len(sys.argv) != 2:
        print("❌ 错误: 使用方法: python occupy_single_gpu.py <gpu_id>")
        print("   例如: python occupy_single_gpu.py 0")
        sys.exit(1)
        
    try:
        gpu_id = int(sys.argv[1])
    except ValueError:
        print(f"❌ 错误: GPU ID '{sys.argv[1]}' 必须是一个整数。")
        sys.exit(1)
        
    # 运行占用函数
    occupy_gpu(gpu_id)

if __name__ == "__main__":
    main()