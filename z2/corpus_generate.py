"""
语料生成与存储模块

主要功能：
1. TrafficEmbeddingGenerator: 使用 TrafficEmbedder 生成流量向量
2. TextCorpusGenerator: 使用 ProposeModel 生成文本语料
3. 向量库和 BM25 语料库的构建
"""

import json
import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from z2.RAG.utils import save_corpus

class TrafficEmbeddingGenerator:
    """
    流量嵌入生成器 - 从预处理的数据集生成流量向量
    
    使用 TrafficEmbedder 模型对流量进行编码，生成向量并存储。
    """
    
    def __init__(self, args):
        """
        初始化流量嵌入生成器
        
        Args:
            args: 模型参数，需包含 TrafficEmbedder 所需的配置
            device: 设备 ('cuda' 或 'cpu')，None 则自动选择
        """
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            from z2.model import TrafficEmbedder
            from transformers import AutoProcessor
            
            self.model = TrafficEmbedder(self.args)
            self.model.to(self.device)
            # self.model.dispatch()
            self.model.resume(self.args)
            self.model.eval()
            
            self.processor = AutoProcessor.from_pretrained(self.args.llm)
            
            print(f"✅ TrafficEmbedder 已加载 (设备: {self.device})")
    
    @torch.no_grad()
    def generate_embeddings(
        self,
        dataset_path: str,
        output_dir: str,
        batch_size: int = 8,
        save_threshold: int = 1000,
        normalize: bool = True
    ) -> None:
        """
        从数据集生成流量嵌入向量，使用 CustomDataset 和累积阈值存储
        
        Args:
            dataset_path: 数据集目录路径
            output_dir: 输出目录路径（每批会保存到单独的文件）
            batch_size: 数据处理批处理大小
            save_threshold: 累积多少样本后保存一次
            normalize: 是否归一化嵌入向量
        """
        from torch.utils.data import DataLoader
        from dataset import CustomDataset, collate_TrafficEmbeddingDataset
        
        self._load_model()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        print(f"📂 正在加载数据集: {dataset_path}")
        dataset = CustomDataset(dataset_path)
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_TrafficEmbeddingDataset,
            num_workers=0
        )
        
        print(f"🔄 开始生成嵌入向量...")
        print(f"   - 处理批次大小: {batch_size}")
        print(f"   - 存储阈值: {save_threshold}")
        
        total_samples = 0
        save_batch_idx = 0
        
        # 累积的embeddings和ids
        accumulated_embeddings = []
        accumulated_ids = []
        
        # 分批处理和存储
        for batch_data, txt_filenames in tqdm(dataloader, desc="生成嵌入"):
            # batch_data 是字典: {input_ids, payloads, position_ids, attention_mask}
            # labels 是列表，包含 txt_filename (id)
            
            # 生成嵌入并直接转到cpu
            embeddings = self.model(**batch_data, normalize=normalize).cpu()
            embeddings_np = embeddings.numpy()
            
            # 从 labels 中直接获取 id
            accumulated_ids.extend(txt_filenames)
            
            # 累积embeddings
            accumulated_embeddings.append(embeddings_np)
            
            # 检查是否达到存储阈值
            if len(accumulated_ids) >= save_threshold:
                # 合并并保存
                merged_embeddings = np.vstack(accumulated_embeddings)
                
                batch_output = {
                    'embeddings': merged_embeddings,
                    'ids': accumulated_ids,
                    'dim': merged_embeddings.shape[1],
                    'num_samples': len(accumulated_ids)
                }
                
                batch_file = os.path.join(output_dir, f'embeddings_batch_{save_batch_idx:05d}.pkl')
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_output, f)
                
                total_samples += len(accumulated_ids)
                save_batch_idx += 1
                
                # 重置累积数据
                accumulated_embeddings = []
                accumulated_ids = []
        
        # 保存剩余的数据
        if len(accumulated_ids) > 0:
            merged_embeddings = np.vstack(accumulated_embeddings)
            
            batch_output = {
                'embeddings': merged_embeddings,
                'ids': accumulated_ids,
                'dim': merged_embeddings.shape[1],
                'num_samples': len(accumulated_ids)
            }
            
            batch_file = os.path.join(output_dir, f'embeddings_batch_{save_batch_idx:05d}.pkl')
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_output, f)
            
            total_samples += len(accumulated_ids)
            save_batch_idx += 1
        
        print(f"\n✅ 嵌入生成完成！")
        print(f"   - 样本总数: {total_samples}")
        print(f"   - 保存批次数量: {save_batch_idx}")
        print(f"   - 输出目录: {output_dir}")
    
    #TODO：分批构建索引
    def build_vector_index(
        self,
        embeddings_path: str,
        index_dir: str,
        index_type: str = 'flat',
        verbose: bool = True
    ) -> None:
        """
        从嵌入文件构建 FAISS 向量索引（使用余弦相似度）
        
        Args:
            embeddings_path: 嵌入文件路径 (.pkl)
            index_dir: 索引保存目录
            index_type: 索引类型 ('flat' 或 'ivf')
            verbose: 是否打印详细信息
        """
        from z2.RAG.vector_utils import build_faiss_index
        
        # 遍历输入目录下的所有 pkl 文件
        embeddings_dir = embeddings_path
        pkl_files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.pkl')])
        
        if not pkl_files:
            raise FileNotFoundError(f"目录下没有找到 .pkl 文件: {embeddings_dir}")
        
        all_embeddings = []
        all_ids = []
        
        for pkl_file in pkl_files:
            pkl_path = os.path.join(embeddings_dir, pkl_file)
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            all_embeddings.append(data['embeddings'])
            all_ids.extend(data['ids'])
        
        embeddings = np.vstack(all_embeddings)
        
        # 使用共享的索引构建函数
        build_faiss_index(
            embeddings=embeddings,
            doc_ids=all_ids,
            doc_contents=None,
            index_dir=index_dir,
            index_type=index_type,
            verbose=verbose
        )

def run_traffic_embedding_pipeline(
    # 数据集路径
    dataset_path: str,
    # 输出路径
    embeddings_output_dir: str,
    index_output_dir: str,
    # 模型参数
    llm: str = 'Qwen3-VL-Embedding-2B',
    projector: str = 'linear',
    linear_output_dim: int = 2048,
    # 生成参数
    batch_size: int = 8,
    save_threshold: int = 1000,
    normalize: bool = True,
    # 索引参数
    index_type: str = 'flat',
    verbose: bool = True,
    # 加载参数
    resume_log: bool = True,
    resume_encoder: str = None,
    resume_linear: str = None,
    resume_lora0: str = None
):
    """
    流量嵌入生成流水线：生成嵌入向量 + 构建向量索引
    
    Args:
        dataset_path: 数据集目录路径
        embeddings_output_dir: 嵌入向量输出目录
        index_output_dir: 向量索引输出目录
        llm: LLM 模型路径
        train_mode: 是否为训练模式
        eval_mode: 是否为评估模式
        projector: 投影器类型
        projector_arch: 投影器架构
        adapter_path: 适配器路径（可选）
        batch_size: 批处理大小
        save_threshold: 存储阈值
        normalize: 是否归一化
        index_type: 索引类型 ('flat' 或 'ivf')
        device: 设备
        verbose: 是否打印详细信息
    """
    from types import SimpleNamespace
    
    print("=" * 60)
    print("流量嵌入生成流水线")
    print("=" * 60)
    
    # 构建 args
    args = SimpleNamespace(
        llm=llm,
        projector=projector,
        linear_output_dim=linear_output_dim,
        resume_log=resume_log,
        resume_encoder=resume_encoder,
        resume_linear=resume_linear,
        resume_lora0=resume_lora0,
        eval_mode=True,
        train_mode=False
    )
    
    # 创建生成器
    generator = TrafficEmbeddingGenerator(args)
    
    # 步骤 1: 生成嵌入向量
    print("\n【步骤 1/2】生成流量嵌入向量")
    print("-" * 60)
    generator.generate_embeddings(
        dataset_path=dataset_path,
        output_dir=embeddings_output_dir,
        batch_size=batch_size,
        save_threshold=save_threshold,
        normalize=normalize
    )
    
    # 步骤 2: 构建向量索引
    print("\n【步骤 2/2】构建向量索引")
    print("-" * 60)
    generator.build_vector_index(
        embeddings_path=embeddings_output_dir,
        index_dir=index_output_dir,
        index_type=index_type,
        verbose=verbose
    )
    
    print("\n" + "=" * 60)
    print("✅ 流量嵌入生成流水线完成！")
    print("=" * 60)

class TextCorpusGenerator:
    """
    文本语料生成器 - 从预处理的数据集使用 LLM 生成文本语料
    
    使用 ProposeModel 对流量理解问题进行回答，生成文本语料并存储。
    """
    
    def __init__(self, args, device: str = None):
        """
        初始化文本语料生成器
        
        Args:
            args: 模型参数，需包含 ProposeModel 所需的配置
            device: 设备 ('cuda' 或 'cpu')，None 则自动选择
        """
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            from z1.model import ProposeModel
            from transformers import AutoTokenizer
            
            self.model = ProposeModel(self.args)
            self.model.eval()
            # self.model.to(self.device)
            self.model.dispatch()
            self.model.resume(self.args)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm)
            
            print(f"✅ ProposeModel 已加载 (设备: {self.device})")
    
    @torch.no_grad()
    def generate_corpus(
        self,
        dataset_path: str,
        output_dir: str,
        num_generations: int = 5,
        entropy_threshold: float = 1.5,
        save_threshold: int = 1000,
        max_new_tokens: int = 512,
        min_new_tokens: int = 64,
        temperature: float = 0.7,
        repetition_penalty: float = 1.25,
        generation_mode: str = "batch",
        early_stop_batch: int = None
    ) -> None:
        """
        从数据集生成文本语料，使用多结果生成、LLM聚类和熵筛选
        
        对每个样本：
        1. 生成 num_generations 条结果
        2. 将结果拼接后输入 LLM 进行语义聚类
        3. 根据聚类结果计算熵
        4. 若熵超过阈值则抛弃本轮输出，否则从结果数量最多的聚类中随机选择一条
        
        Args:
            dataset_path: 数据集目录路径
            output_dir: 输出目录路径（每批会保存到单独的文件）
            num_generations: 每个样本生成的结果数量
            entropy_threshold: 熵阈值，超过则抛弃本轮输出
            save_threshold: 累积多少样本后保存一次
            max_new_tokens: 最大生成 token 数
            generation_mode: 生成模式（"batch" 一次性生成 / "loop" 循环逐条生成）
        """
        import math
        import random
        import re
        from torch.utils.data import DataLoader
        from dataset import CustomDataset, collate_LLMDataset_leftpadding
        
        self._load_model()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        print(f"📂 正在加载数据集: {dataset_path}")
        dataset = CustomDataset(dataset_path)
        
        # 创建 DataLoader，batch_size 固定为 1
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda batch: collate_LLMDataset_leftpadding(batch, keep_labels=False),
            num_workers=0
        )

        generation_mode = generation_mode.lower()
        if generation_mode not in {"batch", "loop"}:
            raise ValueError(f"generation_mode 必须是 'batch' 或 'loop'，当前: {generation_mode}")
        
        print(f"🔄 开始生成文本语料...")
        print(f"   - 每样本生成数量: {num_generations}")
        print(f"   - 生成模式: {generation_mode}")
        print(f"   - 熵阈值: {entropy_threshold}")
        print(f"   - 存储阈值: {save_threshold}")
        print(f"   - 最大生成token数: {max_new_tokens}")
        print(f"   - 最小生成token数: {min_new_tokens}")
        print(f"   - 温度: {temperature}")
        print(f"   - 重复惩罚: {repetition_penalty}")
        if early_stop_batch is not None:
            print(f"   - 提前停止批次: {early_stop_batch}")

        
        total_samples = 0
        discarded_samples = 0
        save_batch_idx = 0
        
        # 累积的语料和ids
        accumulated_corpus = []
        accumulated_ids = []
        
        def compute_entropy(cluster_sizes: list) -> float:
            """计算聚类结果的熵"""
            total = sum(cluster_sizes)
            if total == 0:
                return 0.0
            entropy = 0.0
            for size in cluster_sizes:
                if size > 0:
                    p = size / total
                    entropy -= p * math.log2(p)
            return entropy
        
        def parse_clusters(cluster_text: str, num_results: int) -> dict:
            """解析 LLM 输出的聚类结果
            
            期望格式: 
            聚类1: 1, 3, 5
            聚类2: 2, 4
            
            Raises:
                ValueError: 解析失败或聚类结果未覆盖所有编号或有重复编号
            """
            clusters = {}
            # 覆盖状态数组，False表示未覆盖，True表示已覆盖
            covered = [False] * (num_results + 1)  # 索引0不使用，1~num_results对应编号
            uncovered_count = num_results  # 剩余未被覆盖的编号数量
            
            # 匹配 "聚类X: 1, 2, 3" 或 "类别X: 1, 2, 3" 格式
            pattern = r'(?:聚类|类别|Cluster)\s*(\d+)\s*[:：]\s*([\d,，\s]+)'
            matches = re.findall(pattern, cluster_text, re.IGNORECASE)
            
            for cluster_id, members_str in matches:
                # 解析成员编号
                members_str = members_str.replace('，', ',')
                members = []
                for m in members_str.split(','):
                    m = m.strip()
                    if m.isdigit():
                        idx = int(m)
                        if 1 <= idx <= num_results:
                            if not covered[idx]:
                                covered[idx] = True
                                uncovered_count -= 1
                                members.append(idx)
                            else:
                                raise ValueError(f"聚类解析失败: 编号 {idx} 重复出现")
                if members:
                    clusters[int(cluster_id)] = members
            
            # 如果解析失败，抛出错误
            if not clusters:
                raise ValueError(f"聚类解析失败: 未能从输出中提取有效聚类")
            
            # 检查是否覆盖了所有编号
            if uncovered_count != 0:
                missing = [i for i in range(1, num_results + 1) if not covered[i]]
                raise ValueError(f"聚类结果不完整: 缺少编号 {missing}")
            
            return clusters
        
        def build_clustering_prompt(results: list) -> str:
            """构建聚类 prompt"""
            results_text = "\n".join([
                f"[{i+1}] {r}" for i, r in enumerate(results)
            ])
            prompt = f"""<|im_start|>system
你是一个文本聚类助手，擅长从语义角度对文本进行聚类。<|im_end|>
<|im_start|>user
以下是针对同一网络流量的多条分析结果，请从语义角度将它们聚类，直接输出聚类结果不需要输出解释。

分析结果：
{results_text}

请按以下格式输出聚类结果，每行一个聚类，包含属于该聚类的结果编号：
聚类1: 1, 3, 5
聚类2: 2, 4
...<|im_end|>
<|im_start|>assistant
"""
            return prompt
        
        # 分批处理和存储
        i = 0
        for batch_data, txt_filenames in tqdm(dataloader, desc="生成语料"):
            i += 1
            if early_stop_batch is not None and i > early_stop_batch:
                break
            try:
                # batch_size=1，所以 txt_filenames 只有一个元素
                sample_id = txt_filenames[0]
                input_length = batch_data['input_ids'].shape[1]
                
                if generation_mode == "batch":
                    # 1. 对同一样本生成多条结果（通过复制 batch_data 实现批量生成）
                    expanded_batch = {}
                    for key, value in batch_data.items():
                        if key == 'labels':
                            continue
                        if key == 'payloads':
                            # payloads 是列表，需要复制每个元素
                            assert len(value) == 1 and isinstance(value[0], tuple) and len(value[0]) == 3
                            expanded_batch[key] = [value[0] for _ in range(num_generations)]
                        elif key == 'position_ids':
                            # position_ids 在第二个维度复制，形状从 [3, 1, seq_len] 变成 [3, num_generations, seq_len]
                            expanded_batch[key] = value.repeat(1, num_generations, 1)
                        else:
                            # 其他 tensor 数据，在第一个维度复制
                            expanded_batch[key] = value.repeat(num_generations, 1)

                    # expanded_batch = {
                    #     k: (
                    #         [x.to(self.device) if torch.is_tensor(x) else x for x in v]
                    #         if isinstance(v, list)
                    #         else (v.to(self.device) if torch.is_tensor(v) else v)
                    #     )
                    #     for k, v in expanded_batch.items()
                    # }
                    
                    # 批量生成
                    outputs = self.model.generate(
                        **expanded_batch,
                        max_new_tokens=max_new_tokens,
                        # min_new_tokens=min_new_tokens,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9
                    ).cpu()
                else:
                    # 逐条生成（循环 num_generations 次，每次只处理一条数据）
                    outputs_list = []
                    for _ in range(num_generations):
                        output = self.model.generate(
                            **batch_data,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            repetition_penalty=repetition_penalty,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.9,
                        ).cpu()
                        outputs_list.append(output[0])

                    pad_token_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else 0
                    )
                    outputs = torch.nn.utils.rnn.pad_sequence(
                        outputs_list,
                        batch_first=True,
                        padding_value=pad_token_id
                    )
                
                if early_stop_batch is not None:
                    accumulated_corpus.append({
                        'id': sample_id,
                        'payloads_len': batch_data['payloads'][0][0].shape[0],
                        'position_ids_shape': list(batch_data['position_ids'].shape),
                        'input_ids_shape': list(batch_data['input_ids'].shape),
                        'attention_mask_shape': list(batch_data['attention_mask'].shape),
                        'input_ids_decoded': self.tokenizer.decode(batch_data['input_ids'][0], skip_special_tokens=False),
                        'attention_mask': ' '.join([str(x.item()) for x in batch_data['attention_mask'][0]]),
                        'outputs_shape': list(outputs.shape),
                        'outputs_decoded': [self.tokenizer.decode(outputs[j], skip_special_tokens=False) for j in range(outputs.shape[0])]
                    })
                    continue
                # 解码所有生成的文本
                generated_results = []
                for i in range(num_generations):
                    generated_text = self.tokenizer.decode(
                        outputs[i][input_length:],
                        skip_special_tokens=True
                    )
                    generated_results.append(generated_text)
                
                #TODO：分类任务作为问题时，可以直接分类，不用LLM聚类
                # 2. 构建聚类 prompt 并让 LLM 进行聚类
                clustering_prompt = build_clustering_prompt(generated_results)
                clustering_input = self.tokenizer(
                    clustering_prompt, 
                    return_tensors='pt', 
                    add_special_tokens=False
                )
                clustering_input['position_ids'] = torch.arange(
                    clustering_input['input_ids'].shape[1]
                ).unsqueeze(0).expand(3, -1, -1)
                clustering_input = {k: v.to(self.device) for k, v in clustering_input.items()}
                
                cluster_output = self.model.generate(
                    **clustering_input,
                    max_new_tokens=64,
                    do_sample=False,
                    temperature=0.1
                ).cpu()
                
                cluster_text = self.tokenizer.decode(
                    cluster_output[0][clustering_input['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # 3. 解析聚类结果
                clusters = parse_clusters(cluster_text, num_generations)
                cluster_sizes = [len(members) for members in clusters.values()]
                
                # 4. 计算熵
                entropy = compute_entropy(cluster_sizes)
                
                # 5. 根据熵决定是否保留结果
                if entropy > entropy_threshold:
                    # 熵超过阈值，抛弃本轮输出
                    discarded_samples += 1
                    continue
                
                # 6. 从结果数量最多的聚类中随机选择一条
                largest_cluster_id = max(clusters.keys(), key=lambda k: len(clusters[k]))
                largest_cluster_members = clusters[largest_cluster_id]
                selected_idx = random.choice(largest_cluster_members) - 1  # 转为 0-indexed
                selected_result = generated_results[selected_idx]
                
                accumulated_corpus.append({
                    'id': sample_id,
                    'contents': selected_result
                })
                accumulated_ids.append(sample_id)
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"Error processing sample {sample_id}: {str(e)}\n详细堆栈信息:\n{error_detail}")
                continue
            
            # 检查是否达到存储阈值
            if len(accumulated_corpus) >= save_threshold:
                # 保存当前批次
                batch_file = os.path.join(output_dir, f'corpus_batch_{save_batch_idx:05d}.jsonl')
                with open(batch_file, 'w', encoding='utf-8') as f:
                    for entry in accumulated_corpus:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                total_samples += len(accumulated_corpus)
                save_batch_idx += 1
                
                # 重置累积数据
                accumulated_corpus = []
                accumulated_ids = []
        
        # 保存剩余的数据
        if len(accumulated_corpus) > 0:
            batch_file = os.path.join(output_dir, f'corpus_batch_{save_batch_idx:05d}.jsonl')
            with open(batch_file, 'w', encoding='utf-8') as f:
                for entry in accumulated_corpus:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            total_samples += len(accumulated_corpus)
            save_batch_idx += 1
        
        print(f"\n✅ 语料生成完成！")
        print(f"   - 有效样本数: {total_samples}")
        print(f"   - 抛弃样本数: {discarded_samples} (熵超过阈值)")
        print(f"   - 保存批次数量: {save_batch_idx}")
        print(f"   - 输出目录: {output_dir}")
    
    def build_bm25_index(
        self,
        corpus_path: str,
        index_dir: str,
        analyzer_name: str = 'whitespace',
        verbose: bool = True
    ) -> None:
        """
        从语料文件构建 BM25 索引
        
        Args:
            corpus_path: 语料文件路径 (.jsonl)
            index_dir: 索引保存目录
            analyzer_name: Lucene 分析器名称
            verbose: 是否打印详细信息
        """
        from z2.RAG.retriever.BM25 import build_index
        
        build_index(
            corpus_file=corpus_path,
            index_dir=index_dir,
            analyzer_name=analyzer_name,
            verbose=verbose
        )

def run_text_corpus_pipeline(
    # 数据集路径
    dataset_path: str,
    # 输出路径
    corpus_output_dir: str,
    index_output_dir: str,
    # 模型参数
    llm: str = 'Qwen3-VL-8B-Instruct',
    projector: str = 'linear',
    linear_output_dim: int = 4096,
    # 生成参数
    num_generations: int = 5,
    entropy_threshold: float = 1.5,
    save_threshold: int = 1000,
    max_new_tokens: int = 512,
    generation_mode: str = "batch",
    # 索引参数
    analyzer_name: str = 'whitespace',
    verbose: bool = True,
    # 加载参数
    resume_log: bool = True,
    resume_encoder: str = None,
    resume_linear: str = None,
    resume_lora0: str = None,
    resume_lora1: str = None,
    # 其他参数
    early_stop_batch: int = None
):
    """
    文本语料生成流水线：生成语料 + 构建 BM25 索引
    
    Args:
        dataset_path: 数据集目录路径
        corpus_output_dir: 语料输出目录
        index_output_dir: BM25 索引输出目录
        llm: LLM 模型路径
        train_mode: 是否为训练模式
        eval_mode: 是否为评估模式
        adapter_path: 适配器路径（可选）
        num_generations: 每个样本生成的结果数量
        entropy_threshold: 熵阈值
        save_threshold: 存储阈值
        max_new_tokens: 最大生成 token 数
        generation_mode: 生成模式（"batch" 一次性生成 / "loop" 循环逐条生成）
        analyzer_name: Lucene 分析器名称
        device: 设备
        verbose: 是否打印详细信息
    """
    from types import SimpleNamespace
    
    print("=" * 60)
    print("文本语料生成流水线")
    print("=" * 60)
    
    # 构建 args
    args = SimpleNamespace(
        llm=llm,
        linear_output_dim=linear_output_dim,
        resume_log=resume_log,
        resume_encoder=resume_encoder,
        resume_linear=resume_linear,
        resume_lora0=resume_lora0,
        resume_lora1=resume_lora1,
        align1_mode=False, align2_mode=False, test_mode=False, eval_mode=True,
        finetune_mode=False,
        projector=projector
    )
    
    # 创建生成器
    generator = TextCorpusGenerator(args)
    
    # 步骤 1: 生成文本语料
    print("\n【步骤 1/2】生成文本语料")
    print("-" * 60)
    generator.generate_corpus(
        dataset_path=dataset_path,
        output_dir=corpus_output_dir,
        num_generations=num_generations,
        entropy_threshold=entropy_threshold,
        save_threshold=save_threshold,
        max_new_tokens=max_new_tokens,
        generation_mode=generation_mode,
        early_stop_batch=early_stop_batch
    )

    if early_stop_batch is not None:
        return
    
    # 步骤 2: 构建 BM25 索引
    print("\n【步骤 2/2】构建 BM25 索引")
    print("-" * 60)
    generator.build_bm25_index(
        corpus_path=corpus_output_dir,
        index_dir=index_output_dir,
        analyzer_name=analyzer_name,
        verbose=verbose
    )
    
    print("\n" + "=" * 60)
    print("✅ 文本语料生成流水线完成！")
    print("=" * 60)


if __name__ == "__main__":
    import fire
    fire.Fire()
