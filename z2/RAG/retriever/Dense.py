import json
import os
import numpy as np
import torch
import faiss
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from z2.RAG.utils import save_corpus


class DenseRetriever:
    """
    基于 FAISS 的稠密向量检索器。
    
    使用 TrafficEmbedder 模型对文档进行编码，构建 FAISS 索引进行高效检索。
    
    Example:
        >>> retriever = DenseRetriever(args)
        >>> retriever.save_corpus([{'id': 'doc1', 'contents': '文档内容'}], 'corpus.jsonl')
        >>> retriever.build_index('corpus.jsonl', 'index_dir')
        >>> results = retriever.search('查询文本', 'index_dir', k=5)
    """
    
    def __init__(self, args, device: str = None):
        """
        初始化检索器。
        
        Args:
            args: 模型参数，需包含 TrafficEmbedder 所需的配置
            device: 设备 ('cuda' 或 'cpu')，None 则自动选择
        """
        self.args = args
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            from z2.model import TrafficEmbedder
            from transformers import AutoProcessor
            
            self.model = TrafficEmbedder(self.args)
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = AutoProcessor.from_pretrained(self.args.llm)
            
            print(f"✅ 模型已加载: TrafficEmbedder (设备: {self.device})")
    
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8, normalize: bool = True) -> np.ndarray:
        """
        对文本列表进行编码，返回向量。
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否对向量进行 L2 归一化
        
        Returns:
            np.ndarray: 形状为 (len(texts), hidden_dim) 的向量数组
        """
        self._load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            embeddings = self.model(**inputs, normalize=normalize)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def build_index(
        self,
        corpus_file: str,
        index_dir: str,
        batch_size: int = 8,
        index_type: str = 'flat',
        verbose: bool = True
    ) -> None:
        """
        从语料文件构建 FAISS 索引。
        
        Args:
            corpus_file: 语料文件路径 (.jsonl)
            index_dir: 索引保存目录
            batch_size: 编码时的批处理大小
            index_type: 索引类型 ('flat' 或 'ivf')
            verbose: 是否打印详细信息
        """
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"语料文件不存在: {corpus_file}")
        
        os.makedirs(index_dir, exist_ok=True)
        
        # 读取语料
        doc_ids = []
        doc_contents = []
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_ids.append(doc['id'])
                doc_contents.append(doc['contents'])
        
        if verbose:
            print(f"📄 已加载 {len(doc_ids)} 个文档")
        
        # 编码文档
        if verbose:
            print("🔄 正在编码文档...")
        embeddings = self.encode(doc_contents, batch_size=batch_size)
        
        # 使用共享的索引构建函数
        from z2.RAG.vector_utils import build_faiss_index
        
        build_faiss_index(
            embeddings=embeddings,
            doc_ids=doc_ids,
            doc_contents=doc_contents,
            index_dir=index_dir,
            index_type=index_type,
            verbose=verbose
        )
    
    def search(
        self,
        query: str,
        index_dir: str,
        k: int = 10,
        return_contents: bool = True,
        verbose: bool = False
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        使用稠密向量进行 top-k 检索。
        
        Args:
            query: 查询文本
            index_dir: 索引目录路径
            k: 返回 top-k 结果数量
            return_contents: 是否返回文档内容
            verbose: 是否打印详细信息
        
        Returns:
            List[Tuple[str, float, Optional[str]]]: (doc_id, score, contents) 列表
        """
        # 编码查询
        query_embedding = self.encode([query], batch_size=1)
        
        # 使用共享的检索函数
        from z2.RAG.vector_utils import search_faiss_index
        
        results = search_faiss_index(
            query_embedding=query_embedding,
            index_dir=index_dir,
            k=k,
            return_contents=return_contents
        )
        
        # 打印详细信息
        if verbose:
            for i, (doc_id, score, content) in enumerate(results, 1):
                print(f"\n排名 {i}: {doc_id} (分数: {score:.4f})")
                if content:
                    print(f"内容: {content[:200]}..." if len(content) > 200 else f"内容: {content}")
            print(f"\n✅ 检索完成，返回 {len(results)} 个结果")
        
        return results


# 便捷函数，兼容 BM25 风格的接口
_default_retriever = None

def _get_retriever(args=None) -> DenseRetriever:
    """获取或创建默认检索器"""
    global _default_retriever
    if _default_retriever is None:
        if args is None:
            raise ValueError("首次调用需要传入 args 参数")
        _default_retriever = DenseRetriever(args)
    return _default_retriever


def build_index(
    args,
    corpus_file: str,
    index_dir: str = 'dense_index',
    batch_size: int = 8,
    verbose: bool = True
) -> None:
    """从语料文件构建 FAISS 索引"""
    retriever = _get_retriever(args)
    retriever.build_index(corpus_file, index_dir, batch_size, verbose=verbose)


def search(
    args,
    query: str,
    index_dir: str = 'dense_index',
    k: int = 10,
    return_contents: bool = True,
    verbose: bool = False
) -> List[Tuple[str, float, Optional[str]]]:
    """使用稠密向量进行 top-k 检索"""
    retriever = _get_retriever(args)
    return retriever.search(query, index_dir, k, return_contents, verbose)