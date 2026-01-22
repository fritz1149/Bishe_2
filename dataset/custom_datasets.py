from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    自定义数据集，用于加载 generate_custom_dataset 生成的数据。
    
    数据格式：
        每个样本为字典: {"lines": [payload1, payload2], "label": int}
        - lines: 包含两个 payload 字符串的列表
        - label: 类别标签 (1=同流不同burst, 2=同burst内, 3=同流不同burst)
    
    目录结构：
        data_path/
        ├── part_00000.pkl
        ├── part_00001.pkl
        └── ...
    
    Args:
        data_path: 数据目录路径（包含 .pkl 文件）
        lazy_load: 是否使用延迟加载模式（不将所有数据读入内存）
    
    使用示例：
        >>> train_dataset = CustomDataset("data/contrastive/train", lazy_load=True)
        >>> train_dataset.randomize()  # 打乱文件顺序
        >>> sample = train_dataset[0]
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise ValueError(f"数据路径不存在: {self.data_path}")
        
        # 加载 id2label.json（如存在）
        self.id2label = self._load_id2label()
        self.samples = self._load_all_samples()
    
    def _load_id2label(self):
        """加载 id2label.json（如存在）"""
        import json
        id2label_path = self.data_path / "id2label.json"
        if id2label_path.exists():
            try:
                with open(id2label_path, "r", encoding="utf-8") as f:
                    id2label = json.load(f)
                print(f"已加载 id2label.json（{len(id2label)} 个标签）")
                return id2label
            except Exception as e:
                print(f"加载 id2label.json 时出错: {e}")
        else:
            print(f"未找到 id2label.json（可选）")
            return None
    
    def _load_all_samples(self) -> List[Dict[str, Any]]:
        """
        从目录中加载所有 pickle 文件的样本。
        
        Returns:
            所有样本的列表
        """
        samples = []
        
        pkl_files = sorted(self.data_path.glob("*.pkl"))
        
        if not pkl_files:
            raise ValueError(f"目录 {self.data_path} 中没有找到 .pkl 文件")
        
        print(f"正在从 {len(pkl_files)} 个文件中加载数据...")
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        print(f"警告: {pkl_file} 不是列表格式，跳过")
            except Exception as e:
                print(f"加载文件 {pkl_file} 时出错: {e}")
                continue
        
        print(f"成功加载 {len(samples)} 个样本")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        sample = self.samples[index]
        return sample["data"], sample["label"]
    
    def __iter__(self):
        """
        支持对数据集进行迭代。
        Yields:
            (data, label)：与 __getitem__ 返回相同，已应用 transform 和 label_transform
        """
        for idx in range(len(self)):
            yield self[idx]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """
        获取数据集中各个标签的分布。
        注意：此方法需要遍历所有数据，延迟加载模式下可能较慢。
        
        Returns:
            {label: count} 字典
        """
        from collections import Counter
        if self.samples is not None and not self.randomized:
            labels = [sample["label"] for sample in self.samples]
        else:
            labels = [self[i][1] for i in range(len(self))]
        return dict(Counter(labels))
    
    def get_raw_sample(self, index: int) -> Dict[str, Any]:
        """
        获取原始样本（未经转换）。
        
        Args:
            index: 样本索引
            
        Returns:
            原始样本字典
        """
        return self.samples[index]


class LMDBDataset(Dataset):
    """
    使用LMDB后端的自定义数据集，适用于大规模数据的高效随机访问。
    
    数据格式：
        每个样本为字典: {"data": ..., "label": int}
    
    LMDB文件结构：
        data_path/
        ├── data.lmdb/        # LMDB数据库目录
        │   ├── data.mdb
        │   └── lock.mdb
        └── id2label.json     # 可选的标签映射
    
    Args:
        data_path: 数据目录路径（包含 data.lmdb 目录）
        readonly: 是否以只读模式打开（默认True）
        lock: 是否使用文件锁（默认False，多进程读取时设为False）
    
    使用示例：
        >>> train_dataset = LMDBDataset("data/contrastive/train")
        >>> sample = train_dataset[0]
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        readonly: bool = True,
        lock: bool = False,
    ):
        super().__init__()
        import lmdb
        
        self.data_path = Path(data_path)
        self.lmdb_path = self.data_path / "data.lmdb"
        
        if not self.lmdb_path.exists():
            raise ValueError(f"LMDB路径不存在: {self.lmdb_path}")
        
        # 加载 id2label.json（如存在）
        self.id2label = self._load_id2label()
        
        # 打开LMDB环境
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=readonly,
            lock=lock,
            readahead=False,
            meminit=False,
        )
        
        # 获取数据集大小
        with self.env.begin(write=False) as txn:
            self._length = txn.stat()["entries"]
        
        print(f"已加载LMDB数据集，共 {self._length} 个样本")
    
    def _load_id2label(self):
        """加载 id2label.json（如存在）"""
        import json
        id2label_path = self.data_path / "id2label.json"
        if id2label_path.exists():
            try:
                with open(id2label_path, "r", encoding="utf-8") as f:
                    id2label = json.load(f)
                print(f"已加载 id2label.json（{len(id2label)} 个标签）")
                return id2label
            except Exception as e:
                print(f"加载 id2label.json 时出错: {e}")
        else:
            print(f"未找到 id2label.json（可选）")
        return None
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        with self.env.begin(write=False) as txn:
            key = str(index).encode("utf-8")
            value = txn.get(key)
            if value is None:
                raise IndexError(f"索引 {index} 不存在")
            sample = pickle.loads(value)
        return sample["data"], sample["label"]
    
    def __iter__(self):
        """
        支持对数据集进行迭代。
        Yields:
            (data, label)
        """
        for idx in range(len(self)):
            yield self[idx]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """
        获取数据集中各个标签的分布。
        
        Returns:
            {label: count} 字典
        """
        from collections import Counter
        labels = [self[i][1] for i in range(len(self))]
        return dict(Counter(labels))
    
    def get_raw_sample(self, index: int) -> Dict[str, Any]:
        """
        获取原始样本（未经转换）。
        
        Args:
            index: 样本索引
            
        Returns:
            原始样本字典
        """
        with self.env.begin(write=False) as txn:
            key = str(index).encode("utf-8")
            value = txn.get(key)
            if value is None:
                raise IndexError(f"索引 {index} 不存在")
            return pickle.loads(value)
    
    def close(self):
        """关闭LMDB环境"""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()

def create_from_pkl_files(
    pkl_dir: Union[str, Path],
    output_dir: Union[str, Path],
    map_size: int = 1024 * 1024 * 1024 * 100,  # 100GB
):
    """
    从pkl文件目录创建LMDB数据库。
    
    Args:
        pkl_dir: 包含pkl文件的目录
        output_dir: 输出目录（将创建data.lmdb子目录）
        map_size: LMDB最大大小（字节），默认100GB
    """
    import lmdb
    import shutil
    
    pkl_dir = Path(pkl_dir)
    output_dir = Path(output_dir)
    lmdb_path = output_dir / "data.lmdb"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果已存在则删除
    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)
    
    # 复制id2label.json（如果存在）
    id2label_src = pkl_dir / "id2label.json"
    if id2label_src.exists():
        shutil.copy(id2label_src, output_dir / "id2label.json")
    
    # 获取所有pkl文件
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"目录 {pkl_dir} 中没有找到 .pkl 文件")
    
    print(f"正在从 {len(pkl_files)} 个pkl文件创建LMDB数据库...")
    
    # 创建LMDB环境
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    
    sample_idx = 0
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    with env.begin(write=True) as txn:
                        for sample in data:
                            key = str(sample_idx).encode("utf-8")
                            value = pickle.dumps(sample)
                            txn.put(key, value)
                            sample_idx += 1
                else:
                    print(f"警告: {pkl_file} 不是列表格式，跳过")
        except Exception as e:
            print(f"处理文件 {pkl_file} 时出错: {e}")
            continue
        
        print(f"  已处理: {pkl_file.name}, 累计样本数: {sample_idx}")
    
    env.close()
    print(f"LMDB数据库创建完成，共 {sample_idx} 个样本，保存至 {lmdb_path}")

# 文件末尾追加
if __name__ == '__main__':
    import fire
    fire.Fire()