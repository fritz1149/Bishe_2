"""
推理数据集生成模块

功能：
1. generate_inference_catalog: 生成推理数据集的catalog（train/val/test划分）
2. generate_inference_dataset: 生成用于推理/评估的数据集（先输出标签再输出解释）
3. generate_fewshot_dataset: 生成few-shot数据集
4. generate_zeroshot_dataset: 生成zero-shot数据集
"""

import os
import random
import gc
import sys
from typing import List, Optional, Dict, Any
from tqdm import tqdm


def generate_inference_catalog(
    preprocess_path: str, 
    dest_path: str, 
    k: int = 500, 
    min_lines: int = 3
):
    """
    从preprocess_path中对每个label读取最多k个文件，生成catalog记录。
    每个label的目录下生成三个txt文件（train.txt, val.txt, test.txt）。

    Args:
        preprocess_path (str): 预处理文件的根目录，目录结构: preprocess_path/label_name/*.txt
        dest_path (str): 保存catalog的目的地目录
        k (int): 每个标签最多采集的文件数量
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        min_lines (int): 文件最少需要的行数
    """
    
    os.makedirs(dest_path, exist_ok=True)

    # 获取所有label子目录
    label_names = [name for name in os.listdir(preprocess_path)
                   if os.path.isdir(os.path.join(preprocess_path, name))]
    
    print(f"发现 {len(label_names)} 个类别: {label_names}")
    
    for label in label_names:
        label_dir = os.path.join(preprocess_path, label)
        if not os.path.isdir(label_dir):
            continue
        
        # 收集.txt文件
        file_list = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if len(file_list) < 10:
            print(f"跳过 {label}：文件数量不足（{len(file_list)} < 10）")
            continue
        
        random.shuffle(file_list)
        
        # 收集有效的pcap文件名
        valid_files = []
        
        for filename in tqdm(file_list, desc=f"处理{label}文件", file=sys.stdout):
            lines = open(os.path.join(label_dir, filename), "r", encoding="utf-8").readlines()
            if len(lines) < min_lines:
                continue
            
            valid_files.append(filename)
            
            if len(valid_files) >= k:
                break

        if len(valid_files) < 10:
            print(f"跳过 {label}：有效文件数量不足（{len(valid_files)} < 10）")
            continue

        # 按比例划分训练集、验证集、测试集
        random.shuffle(valid_files)
        
        # 为当前label创建目录
        label_dest_dir = os.path.join(dest_path, label)
        os.makedirs(label_dest_dir, exist_ok=True)
        
        # 保存txt文件
        with open(os.path.join(label_dest_dir, "test.txt"), "w", encoding="utf-8") as f:
            for filename in valid_files:
                f.write(filename + "\n")
        
        print(f"已生成 {label} 类别的catalog，共 {len(valid_files)} 个文件")
        
        del valid_files
        gc.collect()

    print(f"Catalog生成完成，已保存到: {dest_path}")

def generate_zeroshot_dataset(
    preprocess_path: str, 
    catalog_path: str, 
    dest_path: str, 
    packet_num_in_flow: int = 5
):
    """
    生成zero-shot推理数据集。
    prompt中只给出类别列表，不提供任何示例。

    Args:
        preprocess_path (str): 预处理文件的根目录
        catalog_path (str): catalog文件所在目录
        dest_path (str): 保存数据集的目的地目录
        packet_num_in_flow (int): 每个流包含的包数量
        max_explanation_chars (int): 解释文本最大字符数
    """
    from .utils import _dump_in_chunks, _LM_input, _str_to_ids
    
    os.makedirs(dest_path, exist_ok=True)

    # 获取所有label
    label_names = [name for name in os.listdir(catalog_path)
                   if os.path.isdir(os.path.join(catalog_path, name))]

    # Zero-shot prompt - 不提供任何示例
    prompt = f"""接下来会给出一个流量表格，包含若干个包的头部特征、统计特征和payload。请分析该流量并从以下类别中选择最匹配的类别标签。
可选类别: {", ".join(label_names)}
接下来是流量表格：<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
    
    prompt2 = """<表格结束>
"""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
    
    # 定义样本生成函数，包含长度控制逻辑
    def generate_sample(lines, lines_used, label, label_ids):
        """生成样本并控制长度不超过 4096"""
        lines_used_here = lines_used
        sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
        # 如果样本长度超过 4096，逐步减少使用的行数
        while sample["data"][-1].shape[1] > 4096 and lines_used_here > 1:
            lines_used_here -= 1
            sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
        if sample["data"][-1].shape[1] > 4096:
            raise Exception(f"样本长度始终大于4096，即使只使用最少的行数")
        return sample
    
    all_test_samples = []
    
    for label in label_names:
        label_ids = []  # zero-shot不需要label_ids
        label_dir = os.path.join(preprocess_path, label)
        catalog_label_dir = os.path.join(catalog_path, label)
        catalog_file = os.path.join(catalog_label_dir, "inference.txt")
        
        if not os.path.exists(catalog_file):
            print(f"跳过 {label}: 没有inference.txt")
            continue
        
        with open(catalog_file, "r", encoding="utf-8") as f:
            file_names = [line.strip() for line in f if line.strip()]
        
        samples = []
        for file_name in tqdm(file_names, desc=f"Zero-shot {label}", file=sys.stdout):
            txt_filepath = os.path.join(label_dir, file_name)
            
            if not os.path.exists(txt_filepath):
                continue
                
            lines = open(txt_filepath, "r", encoding="utf-8").readlines()
            assert len(lines) >= 3
            
            try:
                sample = generate_sample(lines, packet_num_in_flow, label, label_ids)
                samples.append(sample)
            except Exception as e:
                print(f"处理 {txt_filepath} 时出错: {e}")
                continue
        
        all_test_samples.extend(samples)
        print(f"  {label}: {len(samples)} 个样本")
        gc.collect()
    
    random.shuffle(all_test_samples)
    _dump_in_chunks(all_test_samples, os.path.join(dest_path, "test"), -1, name="zeroshot_test")
    
    # 保存id2label映射
    import json
    id2label = {str(i): name for i, name in enumerate(sorted(label_names))}
    with open(os.path.join(dest_path, "test", "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    print(f"Zero-shot数据集生成完成，共 {len(all_test_samples)} 个样本，已保存到: {dest_path}")

def generate_fewshot_dataset(
    preprocess_path: str, 
    catalog_path: str, 
    dest_path: str, 
    n_shots: int = 5,
    packet_num_in_flow: int = 5,
    max_explanation_chars: int = 100
):
    """
    生成few-shot推理数据集。
    prompt中提供n_shots个示例，然后要求模型对新样本进行分类。

    Args:
        preprocess_path (str): 预处理文件的根目录
        catalog_path (str): catalog文件所在目录
        dest_path (str): 保存数据集的目的地目录
        n_shots (int): 每个类别提供的示例数量
        packet_num_in_flow (int): 每个流包含的包数量
        max_explanation_chars (int): 解释文本最大字符数
    """
    from .utils import _dump_in_chunks, _LM_input, _str_to_ids, _build_table, _ids_to_str
    
    os.makedirs(os.path.join(dest_path, "test"), exist_ok=True)

    # 获取所有label
    label_names = [name for name in os.listdir(catalog_path)
                   if os.path.isdir(os.path.join(catalog_path, name))]
    
    print(f"Few-shot设置: {n_shots}-shot, 类别数: {len(label_names)}")

    # 首先从训练集中收集few-shot示例
    fewshot_examples = {}
    for label in label_names:
        label_dir = os.path.join(preprocess_path, label)
        catalog_file = os.path.join(catalog_path, label, "train.txt")
        
        if not os.path.exists(catalog_file):
            continue
        
        with open(catalog_file, "r", encoding="utf-8") as f:
            pcap_names = [line.strip() for line in f if line.strip()]
        
        # 随机选择n_shots个示例
        random.shuffle(pcap_names)
        examples = []
        
        for pcap_name in pcap_names[:n_shots * 3]:  # 多选一些以防有无效的
            if len(examples) >= n_shots:
                break
            txt_filename = pcap_name.rsplit('.', 1)[0] + ".txt"
            txt_filepath = os.path.join(label_dir, txt_filename)
            
            if not os.path.exists(txt_filepath):
                continue
            
            lines = open(txt_filepath, "r", encoding="utf-8").readlines()
            if len(lines) < 3:
                continue
            
            examples.append({
                "lines": lines[:packet_num_in_flow],
                "label": label
            })
        
        fewshot_examples[label] = examples
        print(f"  收集 {label} 的 {len(examples)} 个示例")
    
    # 构建few-shot prompt
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行分类和分析。<|im_end|> """
    
    # 构建示例文本
    example_text = "以下是一些分类示例：\n\n"
    for label, examples in fewshot_examples.items():
        for i, ex in enumerate(examples[:n_shots]):
            example_text += f"示例 - 类别 [{label}]:\n"
            example_text += f"（该流量的payload和头部特征表明这是{label}类型的流量）\n\n"
    
    prompt = system_prompt + f"""
<|im_start|>user
{example_text}
现在，请对以下流量进行分类。
请先输出类别标签，然后输出简短的解释（不超过{max_explanation_chars}字）。

可选类别: {", ".join(label_names)}

输出格式（严格遵守）：
<类别标签>|<解释文本>

接下来是待分类的流量表格：<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
    
    prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
"""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
    
    # 生成测试集
    all_test_samples = []
    
    for label in label_names:
        label_text = f"{label}|该流量的特征符合{label}类型的典型模式。<|im_end|>"
        label_ids = _str_to_ids(label_text, type="qwen3vl")[0]
        
        label_dir = os.path.join(preprocess_path, label)
        catalog_file = os.path.join(catalog_path, label, "test.txt")
        
        if not os.path.exists(catalog_file):
            continue
        
        with open(catalog_file, "r", encoding="utf-8") as f:
            pcap_names = [line.strip() for line in f if line.strip()]
        
        # 定义样本生成函数，包含长度控制逻辑
        def generate_sample(lines, lines_used, label, label_ids):
            """生成样本并控制长度不超过 4096"""
            lines_used_here = lines_used
            sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
            # 如果样本长度超过 4096，逐步减少使用的行数
            while sample["data"][-1].shape[1] > 4096 and lines_used_here > 0:
                lines_used_here -= 2
                sample = _LM_input(lines[:lines_used_here], None, None, label_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True)
            if sample["data"][-1].shape[1] > 4096:
                raise Exception(f"样本长度始终大于4096，即使只使用最少的行数")
            return sample
        
        samples = []
        for pcap_name in tqdm(pcap_names, desc=f"Few-shot {label}", file=sys.stdout):
            txt_filename = pcap_name.rsplit('.', 1)[0] + ".txt"
            txt_filepath = os.path.join(label_dir, txt_filename)
            
            if not os.path.exists(txt_filepath):
                continue
            
            lines = open(txt_filepath, "r", encoding="utf-8").readlines()
            if len(lines) < 3:
                continue
            
            try:
                sample = generate_sample(lines, packet_num_in_flow, label, label_ids)
                samples.append(sample)
            except Exception as e:
                print(f"处理 {txt_filepath} 时出错: {e}")
                continue
        
        all_test_samples.extend(samples)
        print(f"  {label}: {len(samples)} 个样本")
        gc.collect()
    
    random.shuffle(all_test_samples)
    _dump_in_chunks(all_test_samples, os.path.join(dest_path, "test"), -1, name=f"fewshot_{n_shots}_test")
    
    # 保存id2label映射
    import json
    id2label = {str(i): name for i, name in enumerate(sorted(label_names))}
    with open(os.path.join(dest_path, "test", "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    print(f"Few-shot ({n_shots}-shot) 数据集生成完成，共 {len(all_test_samples)} 个样本，已保存到: {dest_path}")

if __name__ == '__main__':
    from fire import Fire
    Fire()
