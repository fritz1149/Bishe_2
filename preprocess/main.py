"""
main.py - 预处理模块的主入口

推荐运行方式：python -m preprocess.main

如果直接运行 python preprocess/main.py，需要确保：
1. 从项目根目录运行
2. 或者设置 PYTHONPATH 环境变量指向项目根目录
"""
import sys
import os

# 如果直接运行此脚本（非包模式），将项目根目录添加到 sys.path
# 这样可以让相对导入工作（通过将 preprocess 目录添加到路径）
if __name__ == "__main__" and not __package__:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_current_dir)
    
    # 将项目根目录添加到 sys.path（用于导入 uer 等外部模块）
    if _root_dir not in sys.path:
        sys.path.insert(0, _root_dir)
    
    # 注意：相对导入在直接运行脚本时可能失败
    # 如果失败，请使用 python -m preprocess.main 运行

from .tmp import generate_classify_tmp
from .preprocess import process_flow_dataset
from .utils import check_tmp
from .generate_dataset import generate_contrastive_dataset, generate_alignment_dataset, generate_finetuning_dataset
from fire import Fire

def test():
    from uer.uer.utils import CLS_TOKEN, PAD_TOKEN
    print(CLS_TOKEN)
    print(PAD_TOKEN)

if __name__ == "__main__":
    Fire()