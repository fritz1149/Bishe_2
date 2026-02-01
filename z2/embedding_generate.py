"""
流量嵌入向量生成模块

主要功能：
1. TrafficEmbeddingGenerator: 使用 TrafficEmbedder 生成流量向量
2. 向量库的构建
"""

import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


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


if __name__ == "__main__":
    import fire
    fire.Fire()
