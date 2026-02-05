"""
RAG 系统的数据集生成模块

包含三个主要功能：
1. generate_rag_catalog: 生成 catalog 文件，不区分 label 或 test/train/val
2. generate_embedding_dataset: 生成流量嵌入的数据集
3. generate_corpus_dataset: 生成语料的数据集
"""

import os
import random
import gc
from typing import List, Dict
from tqdm import tqdm


def generate_rag_catalog(preprocess_path: str, dest_path: str, k: int = 500):
    """
    生成 RAG 系统的 catalog 文件，不区分 label 或 test/train/val。
    
    从 preprocess_path 中读取所有 label 的文件，打乱后从中抽取 k 个文件，
    将所有有效的 pcap 文件名保存到一个统一的 catalog.txt 文件中。
    每行格式：label\tpcap_name
    
    Args:
        preprocess_path: 预处理文件的根目录，目录结构: preprocess_path/label_name/*.txt
        dest_path: 保存 catalog 的目的地目录
        k: 总共采集的文件数量
    """
    import sys
    
    os.makedirs(dest_path, exist_ok=True)
    
    # 获取所有 label 子目录
    label_names = [name for name in os.listdir(preprocess_path)
                   if os.path.isdir(os.path.join(preprocess_path, name))]
    
    # 收集所有标签下的txt文件名
    all_files = []  # [(label, filename), ...]
    for label in label_names:
        label_dir = os.path.join(preprocess_path, label)
        if not os.path.isdir(label_dir):
            continue
        
        file_list = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        for filename in file_list:
            all_files.append((label, filename))
    
    print(f"📁 共收集到 {len(all_files)} 个文件")
    
    # 打乱后抽取 k 个
    random.shuffle(all_files)
    catalog_entries = []
    
    for label, filename in tqdm(all_files, desc="处理文件", file=sys.stdout):
        if len(catalog_entries) >= k:
            break
            
        label_dir = os.path.join(preprocess_path, label)
        
        # 检查文件有效性
        try:
            lines = open(os.path.join(label_dir, filename), "r", encoding="utf-8").readlines()
            if len(lines) < 3:
                continue
        except:
            continue
        
        # 保存 label 和 pcap 名称
        catalog_entries.append(f"{label}\t{filename}")
    
    # 保存统一的 catalog 文件
    catalog_path = os.path.join(dest_path, "catalog.txt")
    with open(catalog_path, "w", encoding="utf-8") as f:
        for entry in catalog_entries:
            f.write(entry + "\n")
    
    print(f"\n✅ Catalog 生成完成！")
    print(f"   - 总文件数: {len(catalog_entries)}")
    print(f"   - 保存路径: {catalog_path}")
    
    # 清理内存
    del catalog_entries
    del all_files
    gc.collect()


def generate_embedding_dataset(
    preprocess_path: str,
    catalog_path: str,
    dest_path: str,
    packet_num_in_flow: int = 5
):
    """
    根据 catalog 生成流量嵌入的数据集。
    
    Prompt 设置与 generate_contrastive_dataset_2 一致，即对比学习的格式。
    不区分 label 或 test/train/val，所有数据保存在同一个目录下。
    每个样本保留 pcap 信息作为标识。
    
    Args:
        preprocess_path: 预处理文件的根目录
        catalog_path: catalog 文件所在目录，包含 catalog.txt
        dest_path: 保存数据集的目的地目录
        packet_num_in_flow: 每个流包含的包数量
    """
    from .utils import _LM_input, _str_to_ids, _dump_in_chunks
    import pickle
    import sys
    
    os.makedirs(dest_path, exist_ok=True)
    
    # 读取 catalog
    catalog_file = os.path.join(catalog_path, "catalog.txt")
    if not os.path.exists(catalog_file):
        raise FileNotFoundError(f"Catalog 文件不存在: {catalog_file}")
    
    catalog_entries = []
    with open(catalog_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                label, txt_filename = parts
                catalog_entries.append((label, txt_filename)) 
    print(f"📖 已加载 catalog，共 {len(catalog_entries)} 个条目")
    
    # 准备 prompt（与 generate_contrastive_dataset_2 一致）
    system_prompt = """<|im_start|>system
Represent the user's input.<|im_end|> """
    prompt = system_prompt + f"""
<|im_start|>user
<表格开始>"""
    prompt_ids = _str_to_ids(prompt, type="qwen3vl-emb")[0]
    prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
"""
    prompt2_ids = _str_to_ids(prompt2, type="qwen3vl-emb")[0]
    
    # 生成数据集
    samples = []
    
    # 定义样本生成函数，包含长度控制逻辑
    def generate_sample(lines, lines_used, label):
        """生成样本并控制长度不超过 4096"""
        lines_used_here = lines_used
        sample = _LM_input(lines[:lines_used_here], None, None, [], prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl-emb")
        # 如果样本长度超过 4096，逐步减少使用的行数
        while sample["data"][-1].shape[1] > 4096 and lines_used_here > 0:
            lines_used_here -= 2
            sample = _LM_input(lines[:lines_used_here], None, None, [], prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl-emb")
        if sample["data"][-1].shape[1] > 4096:
            raise Exception(f"样本长度始终大于4096，即使只使用最少的行数")
        return sample
    
    for label, txt_filename in tqdm(catalog_entries, desc="生成嵌入数据集"):
        # 构造文件路径
        txt_path = os.path.join(preprocess_path, label, txt_filename)
        
        if not os.path.exists(txt_path):
            continue
        
        try:
            lines = open(txt_path, "r", encoding="utf-8").readlines()
            assert len(lines) >= 3, f"文件行数不足: {txt_path}"
            
            # 使用长度控制逻辑生成样本
            sample = generate_sample(lines, packet_num_in_flow, f'{label}/{txt_filename}')
            samples.append(sample)
            
        except Exception as e:
            print(f"处理 {txt_path} 时出错: {e}")
            continue
    
    # 保存数据集
    print(f"\n💾 正在保存数据集...")
    _dump_in_chunks(samples, dest_path, chunk_size=1000, name="embedding")
    
    print(f"\n✅ 嵌入数据集生成完成！")
    print(f"   - 样本总数: {len(samples)}")
    print(f"   - 保存路径: {dest_path}")
    
    del samples
    gc.collect()

def generate_corpus_dataset(
    preprocess_path: str,
    catalog_path: str,
    dest_path: str,
    packet_num_in_flow: int = 5,
    understanding_prompts: List[str] = None,
    common_prompt: bool = True
):
    """
    根据 catalog 生成语料的数据集。
    
    Prompt 是硬编码的流量理解问题，流量表征使用 _LM_input 的逻辑。
    不区分 label 或 test/train/val，所有数据保存在同一个目录下。
    每个样本保留 pcap 信息作为标识。
    
    Args:
        preprocess_path: 预处理文件的根目录
        catalog_path: catalog 文件所在目录，包含 catalog.txt
        dest_path: 保存数据集的目的地目录
        packet_num_in_flow: 每个流包含的包数量
        understanding_prompts: 流量理解问题列表，如果为 None 则使用默认问题
    """
    from .utils import _LM_input, _str_to_ids, _dump_in_chunks
    import pickle
    import sys
    
    os.makedirs(dest_path, exist_ok=True)
    
    # 读取 catalog
    catalog_file = os.path.join(catalog_path, "catalog.txt")
    if not os.path.exists(catalog_file):
        raise FileNotFoundError(f"Catalog 文件不存在: {catalog_file}")
    
    catalog_entries = []
    with open(catalog_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                label, txt_filename = parts
                catalog_entries.append((label, txt_filename))
    
    print(f"📖 已加载 catalog，共 {len(catalog_entries)} 个条目")
    
    # 默认的流量理解问题（硬编码，目前留空待完善）
    if common_prompt:
        understanding_prompts = [
            "请分析加密流量中体现出的通信行为模式，并描述该行为可能对应哪种类型的网络活动。请说明判断的依据。",
            "从这段加密流量中，你能推断出通信双方的交互意图吗？请描述其可能的交互逻辑，并说明判断的依据。",
            "这段流量是否表现出正常网络通信的特征？请说明判断的依据。",
            "请描述这段加密流量所展现的会话结构，并解释其结构特点可能反映了什么。请说明判断的依据。",
            "在这段流量中，能否判断哪一端更可能是客户端，哪一端更可能是服务端？请说明判断的依据。"
        ]
    else:
        understanding_prompts = [
            "接下来会给出一个流量表格，包含若干个包的头部特征和统计特征，以及在最后一列的payload。请输出对应的类别。类别包含:sendAudio, sendImage, sendText, shareLocationOnce, transferFile。"
        ]
    
    # 准备 prompt
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|>
<|im_start|>user
"""
    
    # 生成数据集
    samples = []
    
    # 定义样本生成函数，包含长度控制逻辑
    def generate_sample(lines, lines_used, label, prompt_ids, prompt2_ids, answer_ids):
        """生成样本并控制长度不超过 4096"""
        lines_used_here = lines_used
        sample = _LM_input(lines[:lines_used_here], None, None, answer_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl")
        # 如果样本长度超过 4096，逐步减少使用的行数
        while sample["data"][-1].shape[1] > 4096 and lines_used_here > 0:
            lines_used_here -= 2
            sample = _LM_input(lines[:lines_used_here], None, None, answer_ids, prompt_ids, prompt2_ids, label=label, extract_payloads_from_lines=True, biased_avoid=True, token_type="qwen3vl")
        if sample["data"][-1].shape[1] > 4096:
            raise Exception(f"样本长度始终大于4096，即使只使用最少的行数")
        return sample
    
    for label, txt_filename in tqdm(catalog_entries, desc="生成语料数据集"):
        # 构造文件路径
        txt_path = os.path.join(preprocess_path, label, txt_filename)
        
        if not os.path.exists(txt_path):
            continue
        
        try:
            lines = open(txt_path, "r", encoding="utf-8").readlines()
            assert len(lines) >= 3, f"文件行数不足: {txt_path}"
            
            # 随机选择一个理解问题
            question_idx = random.randint(0, len(understanding_prompts) - 1)
            question = understanding_prompts[question_idx]
            
            # 构造完整 prompt
            prompt = system_prompt + f"""接下来会给出一个流量表格，包含若干个包的头部特征和统计特征，以及在最后一列的payload。
请就流量表格来回答以下问题：
{question}
回答的字数应在300到500之间。
接下来是流量表格：<表格开始>"""
            prompt_ids = _str_to_ids(prompt, type="qwen3vl")[0]
            prompt2 = """<表格结束><|im_end|>
<|im_start|>assistant
"""
            prompt2_ids = _str_to_ids(prompt2, type="qwen3vl")[0]
            
            # 使用长度控制逻辑生成样本
            sample = generate_sample(lines, packet_num_in_flow, f'{label}/{txt_filename}', prompt_ids, prompt2_ids, [])
            samples.append(sample)
            
        except Exception as e:
            print(f"处理 {txt_path} 时出错: {e}")
            continue
    
    # 保存数据集
    print(f"\n💾 正在保存数据集...")
    _dump_in_chunks(samples, dest_path, chunk_size=1000, name="corpus")
    
    print(f"\n✅ 语料数据集生成完成！")
    print(f"   - 样本总数: {len(samples)}")
    print(f"   - 保存路径: {dest_path}")
    
    del samples
    gc.collect()


def main():
    """主函数 - 演示数据集生成流程"""
    print("=" * 60)
    print("RAG 数据集生成系统")
    print("=" * 60)
    
    # 示例配置
    preprocess_path = "path/to/preprocess"
    catalog_path = "path/to/catalog"
    embedding_dest = "path/to/embedding_dataset"
    corpus_dest = "path/to/corpus_dataset"
    
    # 步骤 1: 生成 catalog
    print("\n【步骤 1/3】生成 Catalog")
    print("-" * 60)
    # generate_rag_catalog(preprocess_path, catalog_path, k=500)
    
    # 步骤 2: 生成嵌入数据集
    print("\n【步骤 2/3】生成嵌入数据集")
    print("-" * 60)
    # generate_embedding_dataset(preprocess_path, catalog_path, embedding_dest)
    
    # 步骤 3: 生成语料数据集
    print("\n【步骤 3/3】生成语料数据集")
    print("-" * 60)
    # generate_corpus_dataset(preprocess_path, catalog_path, corpus_dest)
    
    print("\n" + "=" * 60)
    print("✅ 数据集生成系统演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    from fire import Fire
    Fire()
