"""
推理测试脚本

包含两种推理模式：
1. 复杂推理（RAG模式）：调用 run_rag_pipeline，包含初始检索、迭代检索、最终生成
2. 简单推理：直接使用模型进行推理，无需复杂的检索流程

功能：
- 环境初始化（单显卡模式/模型并行模式）
- 权重加载
- 数据集初始化
- 推理进行
- 指标计算
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from types import SimpleNamespace
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


@dataclass
class InferenceConfig:
    """推理配置"""
    # 模型配置
    llm_retriever: str = "Qwen3-VL-Embedding-2B"
    llm_generator: str = "Qwen3-VL-8B-Instruct"
    projector: str = "linear"
    linear_output_dim_retriever: int = 2048
    linear_output_dim_generator: int = 4096
    projector_arch: str = "512-512-512"
    
    # 通用权重路径
    resume_log: bool = True
    resume_encoder: str = None
    # ProposeModel 权重路径
    resume_linear_0: str = None
    resume_lora0_0: str = None
    # TrafficEmbedder 权重路径（用于RAG模式）
    resume_linear_1: str = None
    resume_lora0_1: str = None
    
    # 数据集配置
    dataset_path: str = None
    batch_size: int = 1  # 推理时通常使用 batch_size=1
    
    # RAG 配置（复杂推理模式）
    # TODO：透传
    vector_index_dir: str = None
    bm25_index_dir: str = None
    initial_top_k: int = 10
    iterative_top_k: int = 1
    max_iterations: int = 5
    enable_iterative: bool = True
    
    # 生成配置
    # TODO：透传；添加参数种类
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: float = 20
    presence_penalty: float = 1.5
    do_sample: bool = True
    think_first: bool = True
    
    # 设备配置
    device: str = None  # None 表示自动选择
    parallel_mode: bool = False  # 是否使用模型并行
    inference_dtype: str = None  # 推理精度: "bf16", "fp16", None(fp32)
    
    # 输出配置
    output_dir: str = "./inference_results"
    save_predictions: bool = True
    verbose: bool = True

    early_stop: Optional[int] = None


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._init_device()
        self.tokenizer = None
        self.generator = None
        self.retriever = None
        
    def _init_device(self) -> str:
        """初始化设备"""
        if self.config.device:
            return self.config.device
        
        if torch.cuda.is_available():
            if self.config.parallel_mode and torch.cuda.device_count() > 1:
                print(f"🖥️ 使用模型并行模式 (可用GPU: {torch.cuda.device_count()})")
                return "cuda:0"  # 主设备
            else:
                print(f"🖥️ 使用单显卡模式 (GPU: {torch.cuda.get_device_name(0)})")
                return "cuda:0"
        else:
            print("⚠️ CUDA 不可用，使用 CPU")
            return "cpu"
    
    def _get_device_map(self) -> Optional[Dict]:
        """获取模型并行的设备映射"""
        if not self.config.parallel_mode or torch.cuda.device_count() <= 1:
            return None
        
        num_gpus = torch.cuda.device_count()
        # 假设模型有36层，均匀分配到多个GPU
        layers_per_gpu = 36 // num_gpus
        
        device_map = {
            "base_model.model.model.language_model.embed_tokens": "cuda:0",
            "base_model.model.model.language_model.norm": f"cuda:{num_gpus-1}",
            "base_model.model.lm_head": f"cuda:{num_gpus-1}",
            "base_model.model.model.visual": "cuda:0",
            "base_model.model.model.language_model.rotary_emb": "cuda:0",
        }
        
        for i in range(36):
            gpu_id = min(i // layers_per_gpu, num_gpus - 1)
            device_map[f"base_model.model.model.language_model.layers.{i}"] = f"cuda:{gpu_id}"
        
        return device_map
    
    def load_tokenizer(self):
        """加载 tokenizer"""
        if self.tokenizer is None:
            print(f"⏳ 正在加载 Tokenizer: {self.config.llm_generator}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_generator)
            print("✅ Tokenizer 已加载")
        return self.tokenizer
    
    def load_generator(self):
        """加载生成模型 (ProposeModel)"""
        if self.generator is not None:
            return self.generator
        
        print(f"⏳ 正在加载 ProposeModel...")
        from z1.model import ProposeModel
        
        # 解析推理精度
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16}
        torch_dtype = dtype_map.get(self.config.inference_dtype, None)
        if torch_dtype:
            print(f"📐 使用 {self.config.inference_dtype} 精度加载模型")
        else:
            print("📐 使用默认精度 (fp32) 加载模型")
        
        args = SimpleNamespace(
            llm=self.config.llm_generator,
            projector=self.config.projector,
            linear_output_dim=self.config.linear_output_dim_generator,
            resume_log=self.config.resume_log,
            resume_encoder=self.config.resume_encoder,
            resume_linear=self.config.resume_linear_0,
            resume_lora0=self.config.resume_lora0_0,
            test_mode=False,
            align1_mode=False,
            align2_mode=False,
            finetune_mode=False,
            eval_mode=True,
            torch_dtype=torch_dtype
        )
        
        self.generator = ProposeModel(args)
        self.generator.resume(args)
        
        # 模型并行或单卡
        if self.config.parallel_mode:
            device_map = self._get_device_map()
            if device_map:
                self.generator.dispatch(device_map)
        else:
            self.generator = self.generator.to(self.device)
        
        self.generator.device = torch.device(self.device)
        self.generator.eval()
        print(f"✅ ProposeModel 已加载 (设备: {self.device})")
        
        return self.generator
    
    def load_retriever(self):
        """加载 RAG 检索器（用于复杂推理模式）"""
        if self.retriever is not None:
            return self.retriever
        
        if not self.config.vector_index_dir or not self.config.bm25_index_dir:
            raise ValueError("复杂推理模式需要指定 vector_index_dir 和 bm25_index_dir")
        
        from z2.retrieve_and_generate import RAGRetriever, RAGConfig
        
        # 构建 embedder_args
        embedder_args = SimpleNamespace(
            llm=self.config.llm_retriever,
            projector=self.config.projector,
            linear_output_dim=self.config.linear_output_dim_retriever,
            resume_log=self.config.resume_log,
            resume_encoder=self.config.resume_encoder,
            resume_linear=self.config.resume_linear_1,
            resume_lora0=self.config.resume_lora0_1,
            eval_mode=True,
            train_mode=False
        )
        
        # 创建 RAG 配置
        rag_config = RAGConfig(
            vector_index_dir=self.config.vector_index_dir,
            bm25_index_dir=self.config.bm25_index_dir,
            initial_top_k=self.config.initial_top_k,
            iterative_top_k=self.config.iterative_top_k,
            max_iterations=self.config.max_iterations
        )
        
        self.retriever = RAGRetriever(
            embedder_args=embedder_args,
            config=rag_config,
            device=self.device
        )
        
        print("✅ RAG 检索器已初始化")
        return self.retriever
    
    def unload_retriever(self):
        """卸载检索器以释放显存"""
        if self.retriever is not None:
            self.retriever.unload_embedder()
            self.retriever.unload_vector_index()
            self.retriever = None
            torch.cuda.empty_cache()
            print("🗑️ RAG 检索器已卸载")
    
    def load_dataset(self) -> DataLoader:
        """加载数据集"""
        if not self.config.dataset_path:
            raise ValueError("需要指定 dataset_path")
        
        from dataset import CustomDataset, collate_LLMDataset_leftpadding
        
        print(f"📂 正在加载数据集: {self.config.dataset_path}")
        dataset = CustomDataset(self.config.dataset_path)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_LLMDataset_leftpadding(batch, keep_labels=False),
            num_workers=0
        )
        
        print(f"✅ 数据集已加载 (样本数: {len(dataset)})")
        return dataloader

class SimpleInference:
    """简单推理模式：直接使用模型进行推理"""
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.config = engine.config
    
    @torch.no_grad()
    def run(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        """
        执行简单推理
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            预测结果列表
        """
        self.engine.load_tokenizer()
        self.engine.load_generator()
        
        results = []
        device = self.engine.device

        from z2.retrieve_and_generate import generate_response
        
        print("\n" + "=" * 60)
        print("🚀 开始简单推理")
        print("=" * 60)
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc="推理进度")):
            if self.config.early_stop is not None and batch_idx >= self.config.early_stop:
                break

            batch_data, label = data
            
            generated_text = generate_response(
                generator=self.engine.generator,
                tokenizer=self.engine.tokenizer,
                batch_data=batch_data,
                max_new_tokens=self.config.max_new_tokens,
                think_first=self.config.think_first,
                have_corpus=False,
                corpus_list=[]
            )
                
            result = {
                'batch_idx': batch_idx,
                'sample_idx': batch_idx * self.config.batch_size,
                'generated_text': generated_text,
                'label': label,
            }
            results.append(result)
            
            if self.config.verbose and batch_idx < 3:
                print(f"\n--- 样本 {result['sample_idx']} ---")
                print(f"生成: {generated_text[:200]}...")
        
        print(f"\n✅ 简单推理完成，共 {len(results)} 个样本")
        return results

class ComplexInference:
    """
    复杂推理模式：使用 RAG 流程进行推理
    
    流程分为两个阶段：
    1. 初始检索阶段：遍历所有样本，使用 TrafficEmbedder 执行向量检索，记录每个样本的初始检索结果
    2. 后续推理阶段：卸载 embedder 后，针对每个样本动态创建临时索引，执行迭代检索和最终生成
    """
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.config = engine.config
    
    @torch.no_grad()
    def _phase1_initial_retrieval(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        """
        阶段1：初始检索
        
        遍历所有样本，使用 TrafficEmbedder 执行向量检索，记录每个样本的初始检索结果ID
        
        Returns:
            每个样本的初始检索信息列表，包含 batch_data, question, labels, initial_corpus_ids
        """
        from z2.retrieve_and_generate import get_traffic_corelated_corpus
        
        self.engine.load_tokenizer()
        self.engine.load_retriever()
        
        initial_retrieval_results = []
        
        print("\n" + "=" * 60)
        print("🔍 阶段1：初始检索（使用 TrafficEmbedder）")
        print("=" * 60)
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="初始检索")):
            if self.config.early_stop is not None and batch_idx >= self.config.early_stop:
                break

            batch_data, label = batch_data
            
            try:
                # 执行初始检索
                initial_corpus = get_traffic_corelated_corpus(
                    self.engine.retriever, 
                    batch_data,
                    top_k=self.config.initial_top_k
                )
                
                # 记录初始检索结果的 ID 和内容
                initial_corpus_ids = [c['id'] for c in initial_corpus]
                
                initial_retrieval_results.append({
                    'batch_idx': batch_idx,
                    'batch_data': batch_data,
                    'label': label,
                    'initial_corpus': initial_corpus,
                    'initial_corpus_ids': initial_corpus_ids
                })
                
                if self.config.verbose and batch_idx < 3:
                    print(f"   样本 {batch_idx}: 检索到 {len(initial_corpus)} 个语料")
                    
            except Exception as e:
                import traceback
                print(f"⚠️ 样本 {batch_idx} 初始检索失败: {e}")
                traceback.print_exc()
                initial_retrieval_results.append({
                    'batch_idx': batch_idx,
                    'batch_data': batch_data,
                    'label': label,
                    'initial_corpus': [],
                    'initial_corpus_ids': [],
                    'error': str(e)
                })
        
        print(f"\n✅ 初始检索完成，共处理 {len(initial_retrieval_results)} 个样本")
        return initial_retrieval_results
    
    @torch.no_grad()
    def _phase2_iterative_and_generate(
        self, 
        initial_retrieval_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        阶段2：迭代检索与最终生成
        
        针对每个样本，根据其初始检索结果动态创建临时 BM25 索引，执行迭代检索和最终生成
        
        Args:
            initial_retrieval_results: 阶段1的初始检索结果
        
        Returns:
            最终推理结果列表
        """
        from z2.retrieve_and_generate import (
            TempBM25Index, 
            retrieve_iteratively, 
            generate_response,
            RAGConfig
        )
        
        self.engine.load_generator()
        
        results = []
        
        print("\n" + "=" * 60)
        print("🚀 阶段2：迭代检索与最终生成（使用 ProposeModel）")
        print("=" * 60)
        
        # 创建 RAG 配置
        rag_config = RAGConfig(
            vector_index_dir=self.config.vector_index_dir or "",
            bm25_index_dir=self.config.bm25_index_dir or "",
            initial_top_k=self.config.initial_top_k,
            iterative_top_k=self.config.iterative_top_k,
            max_iterations=self.config.max_iterations
        )
        
        for loop_idx, item in enumerate(tqdm(initial_retrieval_results, desc="推理进度")):
            if self.config.early_stop is not None and loop_idx >= self.config.early_stop:
                break

            batch_idx = item['batch_idx']
            batch_data = item['batch_data']
            initial_corpus = item['initial_corpus']
            label = item['label']
            
            # 检查是否有初始检索错误
            if 'error' in item:
                results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': batch_idx * self.config.batch_size,
                    'error': f"初始检索失败: {item['error']}",
                    'label': label
                })
                continue
            
            try:
                # 迭代检索（如果启用）
                iterative_result = None
                final_corpus = initial_corpus
                
                if self.config.enable_iterative and initial_corpus:
                    # 执行迭代检索（不需要 retriever，因为只使用临时索引）
                    iterative_result = retrieve_iteratively(
                        generator=self.engine.generator,
                        tokenizer=self.engine.tokenizer,
                        batch_data=batch_data,
                        initial_corpus=initial_corpus,
                        config=rag_config
                    )
                    final_corpus = iterative_result['all_corpus']
                
                # 最终生成
                response, original = generate_response(
                    generator=self.engine.generator,
                    tokenizer=self.engine.tokenizer,
                    batch_data=batch_data,
                    corpus_list=final_corpus,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                result = {
                    'batch_idx': batch_idx,
                    'sample_idx': batch_idx * self.config.batch_size,
                    'generated_text': response,
                    'original_text': original,
                    'label': label,
                    'initial_corpus_count': len(initial_corpus),
                    'initial_corpus_ids': item['initial_corpus_ids'],
                    'final_corpus_count': len(final_corpus),
                    'iterations': len(iterative_result['iterations']) if iterative_result else 0
                }
                results.append(result)
                
                if self.config.verbose and batch_idx < 3:
                    print(f"\n--- 样本 {result['sample_idx']} ---")
                    print(f"生成: {response[:200]}...")
                    print(f"初始语料数: {result['initial_corpus_count']}, 最终语料数: {result['final_corpus_count']}")
                
                # 每个样本处理完后清理显存
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️ 样本 {batch_idx} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': batch_idx * self.config.batch_size,
                    'error': str(e),
                    'label': label
                })
                # 出错时也清理显存，避免累积
                torch.cuda.empty_cache()
        
        print(f"\n✅ 推理完成，共 {len(results)} 个样本")
        return results
    
    @torch.no_grad()
    def run(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        """
        执行复杂推理（RAG模式）
        
        分两阶段执行：
        1. 初始检索：遍历所有样本，记录初始检索结果
        2. 卸载 embedder 后，针对每个样本执行迭代检索和最终生成
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            预测结果列表
        """
        print("\n" + "=" * 60)
        print("🚀 开始复杂推理（RAG模式 - 两阶段）")
        print("=" * 60)
        
        # 阶段1：初始检索
        initial_retrieval_results = self._phase1_initial_retrieval(dataloader)
        
        # 卸载 embedder 以释放显存
        print("\n🗑️ 卸载 TrafficEmbedder 以释放显存...")
        self.engine.unload_retriever()
        
        # 阶段2：迭代检索与最终生成
        results = self._phase2_iterative_and_generate(initial_retrieval_results)
        
        print(f"\n✅ 复杂推理完成，共 {len(results)} 个样本")
        return results

class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def parse_prediction(text: str) -> Optional[str]:
        """
        从生成文本中解析预测类别
        
        期望格式：
        类别：[分类标签]
        解释：[解释文本]
        """
        import re
        
        # 仅匹配 "类别：XXX" 格式，匹配到行尾
        pattern = r'类别[：:]\s*(.+?)$'
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # 未匹配到预期格式，抛出错误
        raise ValueError(f"无法解析预测结果，未找到'类别：'格式。原文本: {text[:100]}...")
    
    @staticmethod
    def calculate_metrics(
        results: List[Dict[str, Any]], 
        id2label: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        计算评估指标
        
        Args:
            results: 推理结果列表
            id2label: 标签ID到标签名的映射
        
        Returns:
            指标字典
        """
        predictions = []
        ground_truths = []
        parse_failures = 0
        error_samples = 0
        
        def _get_label2id(id2label: Dict[str, str]) -> Dict[str, int]:
            """从 id2label 计算 label2id"""
            return {v: int(k) for k, v in id2label.items()}

        label2id = _get_label2id(id2label)
        print(f"\n📊 标签映射 (label2id): {_get_label2id(id2label) if id2label else 'None'}")

        
        for result in results:
            label = result.get('label')
            assert label is not None
            # 转换 ground truth
            ground_truths.append(label2id[str(label[0])])

            if 'error' in result:
                predictions.append(len(label2id))
                error_samples += 1
                continue
            
            generated_text = result.get('generated_text', '')
            try:
                pred = MetricsCalculator.parse_prediction(generated_text)
                predictions.append(label2id[pred])
            except Exception as e:
                predictions.append(len(label2id))
                parse_failures += 1
                continue
        
        if not predictions:
            return {
                'error': '无法计算指标：没有有效的预测结果',
                'total_samples': len(results),
                'parse_failures': parse_failures,
                'error_samples': error_samples
            }
        
        # 计算准确率
        # 注意：这里需要处理预测标签和真实标签的匹配问题
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        accuracy = correct / len(predictions)
        
        metrics = {
            'total_samples': len(results),
            'valid_predictions': len(predictions),
            'parse_failures': parse_failures,
            'error_samples': error_samples,
            'accuracy': accuracy,
            'correct_count': correct
        }
        
        # 如果标签数量有限，计算更详细的指标
        unique_labels = list(set(ground_truths))
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                ground_truths, predictions, average='macro', zero_division=0
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        except Exception as e:
            metrics['metrics_error'] = str(e)
        
        return metrics


def run_inference(
    # 模型参数
    mode: str = "simple",
    llm_generator: str = "Qwen3-VL-8B-Instruct",
    llm_retriever: str = "Qwen3-VL-Embedding-2B",
    projector: str = "linear",
    linear_output_dim_generator: int = 4096,
    linear_output_dim_retriever: int = 2048,
    # 加载参数
    resume_log: bool = False,
    resume_encoder: str = None,
    resume_linear_0: str = None,
    resume_lora0_0: str = None,
    resume_linear_1: str = None,
    resume_lora0_1: str = None,
    # 数据集参数
    dataset_path: str = None,
    batch_size: int = 1,
    # RAG配置参数
    vector_index_dir: str = None,
    bm25_index_dir: str = None,
    initial_top_k: int = 10,
    iterative_top_k: int = 1,
    max_iterations: int = 5,
    enable_iterative: bool = True,
    # 生成配置
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    do_sample: bool = True,
    # 设备配置
    device: str = None,
    parallel_mode: bool = False,
    inference_dtype: str = None,
    # 输出配置
    output_dir: str = "./inference_results",
    verbose: bool = True,
    early_stop: Optional[int] = None
):
    """
    运行推理测试
    
    Args:
        mode: 推理模式 ("simple" 或 "complex")
        llm: LLM 模型路径
        dataset_path: 数据集路径
        output_dir: 输出目录
        generator_weights: 生成模型权重路径
        embedder_weights: 嵌入模型权重路径（RAG模式需要）
        vector_index_dir: 向量索引目录（RAG模式需要）
        bm25_index_dir: BM25索引目录（RAG模式需要）
        parallel_mode: 是否使用模型并行
        max_new_tokens: 最大生成token数
        max_samples: 最大样本数（用于调试）
        verbose: 是否打印详细信息
    """
    print("=" * 60)
    print(f"🔬 推理测试 - {mode.upper()} 模式")
    print("=" * 60)
    
    # 创建配置
    config = InferenceConfig(
        llm_generator=llm_generator,
        llm_retriever=llm_retriever,
        projector=projector,
        linear_output_dim_generator=linear_output_dim_generator,
        linear_output_dim_retriever=linear_output_dim_retriever,
        resume_log=resume_log,
        resume_encoder=resume_encoder,
        resume_linear_0=resume_linear_0,
        resume_lora0_0=resume_lora0_0,
        resume_linear_1=resume_linear_1,
        resume_lora0_1=resume_lora0_1,
        dataset_path=dataset_path,
        batch_size=batch_size,
        vector_index_dir=vector_index_dir,
        bm25_index_dir=bm25_index_dir,
        initial_top_k=initial_top_k,
        iterative_top_k=iterative_top_k,
        max_iterations=max_iterations,
        enable_iterative=enable_iterative,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        do_sample=do_sample,
        device=device,
        parallel_mode=parallel_mode,
        inference_dtype=inference_dtype,
        output_dir=output_dir,
        verbose=verbose,
        early_stop=early_stop
    )
    
    # 创建引擎
    engine = InferenceEngine(config)
    
    # 加载数据集
    dataloader = engine.load_dataset()
    
    # 执行推理
    start_time = time.time()
    
    if mode.lower() == "simple":
        inference = SimpleInference(engine)
    elif mode.lower() == "complex":
        inference = ComplexInference(engine)
    else:
        raise ValueError(f"未知的推理模式: {mode}，请使用 'simple' 或 'complex'")
    
    results = inference.run(dataloader)
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    print("\n" + "=" * 60)
    print("📊 计算评估指标")
    print("=" * 60)
    
    id2label = dataloader.dataset.id2label if hasattr(dataloader.dataset, 'id2label') else None
    assert id2label is not None
    
    metrics = MetricsCalculator.calculate_metrics(results, id2label)
    metrics['elapsed_time_seconds'] = elapsed_time
    metrics['samples_per_second'] = len(results) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n📈 评估结果:")
    print(f"   - 总样本数: {metrics.get('total_samples', 'N/A')}")
    print(f"   - 有效预测: {metrics.get('valid_predictions', 'N/A')}")
    print(f"   - 解析失败: {metrics.get('parse_failures', 'N/A')}")
    print(f"   - 准确率: {metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else "")
    print(f"   - F1分数: {metrics.get('f1_score', 'N/A'):.4f}" if 'f1_score' in metrics else "")
    print(f"   - 耗时: {elapsed_time:.2f}s")
    print(f"   - 速度: {metrics['samples_per_second']:.2f} 样本/秒")
    
    # 保存结果
    if config.save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果
        predictions_path = os.path.join(output_dir, f"{mode}_predictions.json")
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 预测结果已保存: {predictions_path}")
        
        # 保存指标
        metrics_path = os.path.join(output_dir, f"{mode}_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"💾 评估指标已保存: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("✅ 推理测试完成")
    print("=" * 60)
    
    # return results, metrics

if __name__ == "__main__":
    import fire
    fire.Fire(run_inference)
