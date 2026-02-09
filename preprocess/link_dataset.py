#!/usr/bin/env python3
"""
从preprocess_path中对每个label读取k个文件，将源文件硬连接到新的路径。
不进行train/test/val划分，直接将所有采样的文件硬链接到dest_path/label/目录下。
"""

import os
import random
import shutil
from tqdm import tqdm
import sys
import gc


def generate_hardlink_catalog(preprocess_path: str, dest_path: str, k: int = 500):
    """
    从preprocess_path中对每个label读取k个文件，将源文件硬连接到新的路径。
    不进行train/test/val划分，直接将所有采样的文件硬链接到dest_path/label/目录下。

    Args:
        preprocess_path (str): 预处理文件的根目录，目录结构: preprocess_path/label_name/*.txt
        dest_path (str): 保存硬链接的目的地目录，每个label会在dest_path/label_name/下创建硬链接
        k (int): 每个标签最多采集的文件数量
    """
    # 确保目标目录存在
    os.makedirs(dest_path, exist_ok=True)

    # 获取所有label子目录（必须是目录）
    label_names = [name for name in os.listdir(preprocess_path)
                   if os.path.isdir(os.path.join(preprocess_path, name))]

    for label in label_names:
        label_dir = os.path.join(preprocess_path, label)
        if not os.path.isdir(label_dir):
            continue

        # 收集.txt文件
        file_list = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not file_list:
            continue

        # 随机采样k个文件
        if len(file_list) < 10:
            continue
        random.shuffle(file_list)
        sampled_files = file_list[:k]

        # 收集有效的pcap文件名
        pcap_names = []

        for filename in tqdm(sampled_files, desc=f"处理{label}文件", file=sys.stdout):
            # 检查文件
            lines = open(os.path.join(label_dir, filename), "r", encoding="utf-8").readlines()
            if len(lines) < 3:
                continue

            pcap_names.append(filename)

        if len(pcap_names) < 10:
            continue

        # 为当前label创建目标目录
        label_dest_dir = os.path.join(dest_path, label)
        os.makedirs(label_dest_dir, exist_ok=True)

        # 将源文件硬链接到新路径
        linked_count = 0
        for pcap_name in tqdm(pcap_names, desc=f"创建{label}硬链接", file=sys.stdout):
            # 源文件路径
            src_path = os.path.join(label_dir, pcap_name)
            # 目标文件路径
            dst_path = os.path.join(label_dest_dir, pcap_name)

            # 如果目标文件已存在，跳过
            if os.path.exists(dst_path):
                print(f"目标文件已存在: {dst_path}")
                continue

            # 检查源文件是否存在
            if not os.path.exists(src_path):
                print(f"源文件不存在: {src_path}")
                continue

            # 创建硬链接
            try:
                os.link(src_path, dst_path)
                linked_count += 1
            except OSError as e:
                # 如果硬链接失败（可能跨文件系统），尝试复制
                print(f"  警告: 硬链接失败 ({e}), 尝试复制文件...")
                try:
                    shutil.copy2(src_path, dst_path)
                    linked_count += 1
                except Exception as e2:
                    print(f"  错误: 复制文件失败 {dst_path}: {e2}")

        print(f"已为{label}类别创建{linked_count}个硬链接（共{len(pcap_names)}个文件）")

        # 清理内存
        del pcap_names
        gc.collect()

    print(f"每个标签采样{str(k)}个流(txt文件)，已创建硬链接到: {dest_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(generate_hardlink_catalog)

