#!/usr/bin/env python
"""
测试参数解析是否正确

用法:
    python test_args.py
    python test_args.py --no-distributed
    torchrun --nproc_per_node=2 test_args.py
"""

import os
import argparse

def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no-distributed', dest='distributed', 
                       action='store_false', default=True, 
                       help="禁用分布式训练")
    
    args = parser.parse_args()
    
    # 检查环境变量
    rank = os.environ.get('RANK', 'Not set')
    local_rank = os.environ.get('LOCAL_RANK', 'Not set')
    world_size = os.environ.get('WORLD_SIZE', 'Not set')
    
    print("=" * 60)
    print("参数测试")
    print("=" * 60)
    print(f"args.distributed: {args.distributed}")
    print(f"args.batch_size: {args.batch_size}")
    print()
    print("环境变量:")
    print(f"  RANK: {rank}")
    print(f"  LOCAL_RANK: {local_rank}")
    print(f"  WORLD_SIZE: {world_size}")
    print()
    
    # 模拟 init_distributed 逻辑
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if not (ddp and args.distributed):
        print("结果: 单卡模式")
        print("  原因:", end=" ")
        if not ddp:
            print("未检测到 RANK 环境变量（非 torchrun 启动）")
        elif not args.distributed:
            print("--no-distributed 参数禁用了分布式")
    else:
        print(f"结果: 分布式模式 (Rank {rank})")
    
    print("=" * 60)
    print()
    
    # 显示使用建议
    if not ddp:
        print("💡 提示:")
        print("  当前是单进程模式")
        print("  如需测试分布式，使用:")
        print("    torchrun --nproc_per_node=2 test_args.py")
    elif not args.distributed:
        print("⚠️  注意:")
        print("  虽然在 torchrun 环境下，但 --no-distributed 禁用了分布式")
    else:
        print("✅ 分布式配置正确!")

if __name__ == "__main__":
    test_args()









