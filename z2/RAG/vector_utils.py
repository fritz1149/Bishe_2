"""
向量索引工具模块

提供 FAISS 向量索引构建的通用功能，供 corpus_generate.py 和 Dense.py 复用。
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional


def build_faiss_index(
    embeddings: np.ndarray,
    doc_ids: List[str],
    doc_contents: Optional[List[str]],
    index_dir: str,
    index_type: str = 'flat',
    verbose: bool = True
) -> None:
    """
    构建 FAISS 向量索引（使用内积 = 余弦相似度）
    
    Args:
        embeddings: 向量数组，形状为 (num_docs, dim)
        doc_ids: 文档 ID 列表
        doc_contents: 文档内容列表（可选）
        index_dir: 索引保存目录
        index_type: 索引类型 ('flat' 或 'ivf')
        verbose: 是否打印详细信息
    """
    if len(embeddings) != len(doc_ids):
        raise ValueError(f"向量数量 ({len(embeddings)}) 与文档 ID 数量 ({len(doc_ids)}) 不匹配")
    
    if doc_contents and len(doc_contents) != len(doc_ids):
        raise ValueError(f"文档内容数量 ({len(doc_contents)}) 与文档 ID 数量 ({len(doc_ids)}) 不匹配")
    
    dim = embeddings.shape[1]
    
    if verbose:
        print(f"📄 构建索引: {len(doc_ids)} 个文档，维度 {dim}")
    
    # 构建 FAISS 索引（使用内积 = 余弦相似度，前提是向量已归一化）
    if index_type == 'flat':
        index = faiss.IndexFlatIP(dim)  # 内积
    elif index_type == 'ivf':
        nlist = min(100, len(doc_ids) // 10 + 1)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
    
    index.add(embeddings)
    
    # 保存索引和元数据
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, 'index.faiss'))
    
    metadata = {
        'doc_ids': doc_ids,
        'doc_contents': doc_contents if doc_contents else [],
        'index_type': index_type,
        'dim': dim,
        'num_docs': len(doc_ids)
    }
    
    with open(os.path.join(index_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"✅ 索引构建完成！")
        print(f"   - 索引目录: {index_dir}")
        print(f"   - 文档数量: {len(doc_ids)}")
        print(f"   - 向量维度: {dim}")
        print(f"   - 索引类型: {index_type}")


def load_faiss_index(index_dir: str) -> tuple:
    """
    加载 FAISS 索引和元数据
    
    Args:
        index_dir: 索引目录路径
    
    Returns:
        (index, metadata) 元组
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"索引目录不存在: {index_dir}")
    
    index = faiss.read_index(os.path.join(index_dir, 'index.faiss'))
    
    with open(os.path.join(index_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, metadata


def search_faiss_index(
    query_embedding: np.ndarray,
    index_dir: str,
    k: int = 10,
    return_contents: bool = True
) -> List[tuple]:
    """
    在 FAISS 索引中检索
    
    Args:
        query_embedding: 查询向量，形状为 (1, dim) 或 (dim,)
        index_dir: 索引目录路径
        k: 返回 top-k 结果
        return_contents: 是否返回文档内容
    
    Returns:
        [(doc_id, score, content), ...] 列表
    """
    index, metadata = load_faiss_index(index_dir)
    
    # 确保查询向量是 2D
    assert query_embedding.ndim == 2, "查询向量维度必须是2"
    
    # 检索
    scores, indices = index.search(query_embedding, k)
    
    doc_ids = metadata['doc_ids']
    doc_contents = metadata.get('doc_contents', [])
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:  # FAISS 返回 -1 表示没有更多结果
            break
        
        doc_id = doc_ids[idx]
        content = doc_contents[idx] if return_contents and doc_contents else None
        results.append((doc_id, float(score), content))
    
    return results
