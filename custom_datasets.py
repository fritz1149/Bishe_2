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
        transform: 可选的数据转换函数，应用于 payload 对
        label_transform: 可选的标签转换函数
    
    使用示例：
        >>> train_dataset = ContrastiveDataset("data/contrastive/train")
        >>> test_dataset = ContrastiveDataset("data/contrastive/test")
        >>> sample = train_dataset[0]
        >>> print(sample)  # {"lines": [payload1, payload2], "label": 1}
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise ValueError(f"数据路径不存在: {self.data_path}")
        
        # 加载所有 pickle 文件
        self.samples = self._load_all_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到任何样本，请检查路径: {self.data_path}")
    
    def _load_all_samples(self) -> List[Dict[str, Any]]:
        """
        从目录中加载所有 pickle 文件的样本，并尝试加载 id2label.json（如存在）。
        
        Returns:
            所有样本的列表
        """
        import json

        samples = []
        
        # 获取所有 .pkl 文件并排序
        pkl_files = sorted(self.data_path.glob("*.pkl"))
        
        # 新增: 尝试加载 id2label.json
        id2label_path = self.data_path / "id2label.json"
        self.id2label = None
        if id2label_path.exists():
            try:
                with open(id2label_path, "r", encoding="utf-8") as f:
                    self.id2label = json.load(f)
                print(f"已加载 id2label.json（{len(self.id2label)} 个标签）")
            except Exception as e:
                print(f"加载 id2label.json 时出错: {e}")
        else:
            print(f"未找到 id2label.json（可选）")

        if not pkl_files:
            print(f"警告: 目录 {self.data_path} 中没有找到 .pkl 文件")
            return samples
        
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
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        获取指定索引的样本。
        
        Args:
            index: 样本索引
            
        Returns:
            (data, label): 转换后的数据和标签
        """
        return self.samples[index]["data"], self.samples[index]["label"]
    
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
        
        Returns:
            {label: count} 字典
        """
        from collections import Counter
        labels = [sample["label"] for sample in self.samples]
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

def collate_ContrastiveDataset(batch, args):
    x0 = [item[0][0][0] for item in batch]
    x0_mask_len = [item[0][0][1] for item in batch]
    x1 = [item[0][1][0] for item in batch]
    x1_mask_len = [item[0][1][1] for item in batch]
    y = [1 if item[1] == 1 else args.weak_sample_rate for item in batch]

    return torch.tensor(x0), torch.tensor(x0_mask_len), torch.tensor(x1), torch.tensor(x1_mask_len), torch.tensor(y)

#TODO 未完善
def collate_ContrastiveDataset2(batch):
    non_payload_ids1 = [item[0][0] for item in batch]
    payload_ids1 = [item[0][1] for item in batch]
    position_ids1 = [item[0][2] for item in batch]
    non_payload_ids2 = [item[0][3] for item in batch]
    payload_ids2 = [item[0][4] for item in batch]
    position_ids2 = [item[0][5] for item in batch]

    def process(non_payload_ids, payload_ids, position_ids):
        seq_lens = [len(npids)+len(pids)+1 for npids, pids in zip(non_payload_ids, payload_ids)]
        max_seq_len = max(seq_lens)
        input_ids = []
        for npids, pids in zip(non_payload_ids, payload_ids):
            input_seq = npids + pids + [1]
            pad_len = max_seq_len - len(input_seq)
            input_seq += [0] * pad_len
            input_ids.append(input_seq)


    return torch.tensor(non_payload_ids1), torch.tensor(payload_ids1), torch.tensor(position_ids1), torch.tensor(non_payload_ids2), torch.tensor(payload_ids2), torch.tensor(position_ids2)

def collate_ContrastiveDataset_test(batch):
    x = [item[0][0] for item in batch]
    x_mask_len = [item[0][1] for item in batch]
    y = [item[1] for item in batch]

    return torch.tensor(x), torch.tensor(x_mask_len), torch.tensor(y)

def collate_LLMDataset(batch):
    PAD_ID = 151643
    IMAGE_PAD_ID = 151655
    x_ids = [item[0][0] for item in batch]
    y_ids = [item[0][1] for item in batch] # [batch_size, seq_len_sample_1]
    payloads = [item[0][2] for item in batch] # [batch_size, row_num_sample, (1500, 1500, 1500)]
    position_ids = [item[0][3] for item in batch] # [batch_size, 3, total_seq_len_sample_1+total_seq_len_sample_2]
    labels = [item[1] for item in batch] # [batch_size]
    # import sys
    # # print("x_ids_len:", len(x_ids[0]), "y_ids_len:", len(y_ids[0]))
    # sys.stdout.flush()

    for i, item in enumerate(payloads):
        if len(item) == 0:
            payloads[i] = None
            continue
        payload_ids = torch.tensor([x[0] for x in item])
        attention_mask = torch.tensor([x[1] for x in item])
        global_attention_mask = torch.tensor([x[2] for x in item])
        payloads[i] = (payload_ids, attention_mask, global_attention_mask)

    # 计算每个样本的总长度
    seq_lens = [
        len(x_ids) + len(y_ids)
        for x_ids, y_ids in zip(x_ids, y_ids)
    ]
    max_seq_len = max(seq_lens)

    input_ids = []
    target_labels = []
    for x_ids, y_ids in zip(x_ids, y_ids):
        input_seq = x_ids+y_ids
        # 补齐
        pad_len = max_seq_len - len(input_seq)
        input_seq += [PAD_ID] * pad_len
        input_ids.append(input_seq)

        # 构造label，非label部分-100
        label_prefix = [-100] * len(x_ids)
        label_suffix = [-100] * pad_len
        target_labels.append(label_prefix + y_ids + label_suffix)

    input_ids = torch.tensor(input_ids)
    labels_ids = torch.tensor(target_labels)
    # payload_ids = torch.tensor(payload_ids) # payload这里每个样本的序列长度不一样
    for i, position_ids_ in enumerate(position_ids):
        start = position_ids_.max().item()+1
        position_ids[i] = torch.cat([position_ids_, torch.arange(start, start+max_seq_len-position_ids_.shape[1]).unsqueeze(0).expand(3, -1)], dim=1)
    position_ids = torch.stack(position_ids, dim=1)
    attention_mask = (input_ids != PAD_ID).long()
    assert position_ids.shape[0] == 3 and position_ids.shape[1] == input_ids.shape[0] and position_ids.shape[2] == max_seq_len

    # from preprocess.utils import _ids_to_str
    # print("input_ids:",_ids_to_str(input_ids[0], type="qwen3vl"))
    # print("input_ids_len:",len(input_ids[0]))
    # first_not_minus_100 = (labels_ids[0] != -100).nonzero()[0].item()
    # print("input_ids_label_part:", _ids_to_str(input_ids[0][first_not_minus_100:], type="qwen3vl"))
    # print("label_ids_label_part:", _ids_to_str(labels_ids[0][first_not_minus_100:], type="qwen3vl"))
    # sys.stdout.flush()
    
    return input_ids, labels_ids, payloads, position_ids, attention_mask, labels