import json
import os
from typing import List, Dict


def save_corpus(
    corpus: List[Dict[str, str]],
    output_path: str,
    append: bool = True
) -> None:
    """
    保存语料到 jsonl 文件，支持多次调用追加到同一文件。
    
    Args:
        corpus: 语料列表，每个元素是字典，需包含 'id' 和 'contents' 字段
                例如: [{'id': 'doc1', 'contents': '文档内容1'}, ...]
        output_path: 输出文件路径 (.jsonl)
        append: 是否追加模式（默认 True，支持多次调用追加）
    
    Example:
        >>> # 第一次调用
        >>> save_corpus([{'id': 'doc1', 'contents': '文档一'}], 'corpus.jsonl')
        >>> # 第二次调用，追加到同一文件
        >>> save_corpus([{'id': 'doc2', 'contents': '文档二'}], 'corpus.jsonl')
    """
    mode = 'a' if append else 'w'
    
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(output_path, mode, encoding='utf-8') as f:
        for doc in corpus:
            if 'id' not in doc or 'contents' not in doc:
                raise ValueError(f"每个文档必须包含 'id' 和 'contents' 字段，当前文档: {doc}")
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ 已{'追加' if append else '保存'} {len(corpus)} 个文档到 {output_path}")
