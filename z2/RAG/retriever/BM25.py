import json
import os
import subprocess
import sys
from typing import List, Dict, Optional, Tuple, Literal
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import get_lucene_analyzer
from z2.RAG.utils import save_corpus
import jieba


def tokenize_text(text: str, language: Literal['zh', 'en'] = 'zh') -> str:
    """
    对文本进行分词
    
    Args:
        text: 输入文本
        language: 语言选项，'zh' 为中文（使用 jieba），'en' 为英文（空格分割）
    
    Returns:
        分词后的文本，用空格分隔
    """
    if language == 'zh':
        # 中文使用 jieba 分词
        tokens = list(jieba.cut(text))
        # 过滤空白字符
        tokens = [t.strip() for t in tokens if t.strip()]
        return ' '.join(tokens)
    else:
        # 英文直接返回（假设已用空格分隔）
        return text


def build_index(
    corpus_path: str,
    index_dir: str = 'index_dir',
    analyzer_name: str = 'whitespace',
    language: Literal['zh', 'en'] = 'zh',
    verbose: bool = True
) -> None:
    """
    从 corpus_path 目录下的所有 jsonl 文件构建 BM25 索引。
    
    Args:
        corpus_path: 语料目录路径（包含多个 .jsonl 文件）
        index_dir: 索引保存目录
        analyzer_name: Lucene 分析器名称，默认 'whitespace'
                      常用选项: 'whitespace', 'standard', 'english'
        language: 语言选项，'zh' 为中文（使用 jieba 分词），'en' 为英文
        verbose: 是否打印详细信息
    
    Example:
        >>> build_index('corpus_dir/', index_dir='my_index', language='zh')
    """
    if verbose:
        print(f"🔧 构建 BM25 索引")
        print(f"   - 语料目录: {corpus_path}")
        print(f"   - 索引目录: {index_dir}")
        print(f"   - 分析器: {analyzer_name}")
        print(f"   - 语言: {language}")
    import sys
    sys.stdout.flush()
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"语料目录不存在: {corpus_path}")
    
    if os.path.isdir(corpus_path):
        jsonl_files = [f for f in os.listdir(corpus_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            raise FileNotFoundError(f"目录 {corpus_path} 中没有找到 .jsonl 文件")
    elif os.path.isfile(corpus_path):
        if not corpus_path.endswith('.jsonl'):
            raise ValueError(f"语料文件必须是 .jsonl 格式: {corpus_path}")
    else:
        raise FileNotFoundError(f"语料路径不存在: {corpus_path}")

    os.makedirs(index_dir, exist_ok=True)
    args = [
        sys.executable,
        '-m',
        'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', corpus_path,
        '--index', index_dir,
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', '1',
        '--storePositions',
        '--storeDocvectors',
        '--storeRaw',
    ]
    if language:
        args += ['--language', language]

    if verbose:
        print("🚀 调用 Pyserini 一次性构建索引")
        print("   " + " ".join(args))

    subprocess.run(args, check=True)

    if verbose:
        print(f"\n✅ 索引构建完成！")
        print(f"   - 索引目录: {index_dir}")


def search(
    query: str,
    index_dir: str = 'index_dir',
    k: int = 10,
    language: Literal['zh', 'en'] = 'zh',
    return_contents: bool = True,
    verbose: bool = False
) -> List[Tuple[str, float, Optional[str]]]:
    """
    使用 BM25 算法进行 top-k 检索。
    
    Args:
        query: 查询文本
        index_dir: 索引目录路径
        k: 返回 top-k 结果数量
        language: 语言选项，'zh' 为中文（使用 jieba 分词），'en' 为英文
        return_contents: 是否返回文档内容（默认 True）
        verbose: 是否打印详细信息
    
    Returns:
        List[Tuple[str, float, Optional[str]]]: 
        返回列表，每个元素是 (doc_id, score, contents)
        如果 return_contents=False，则 contents 为 None
    
    Example:
        >>> results = search('机器学习深度学习', k=5, language='zh')
        >>> for doc_id, score, content in results:
        ...     print(f"ID: {doc_id}, Score: {score:.4f}")
        ...     print(f"Content: {content[:100]}...\n")
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"索引目录不存在: {index_dir}")
    
    searcher = LuceneSearcher(index_dir)
    
    # 对查询进行分词
    tokenized_query = tokenize_text(query, language)
    
    hits = searcher.search(tokenized_query, k=k)
    
    results = []
    for hit in hits:
        doc_id = hit.docid
        score = hit.score
        contents = hit.lucene_document.get('raw') if return_contents and hit.lucene_document else None
        
        if return_contents and contents:
            try:
                doc_dict = json.loads(contents)
                contents = doc_dict.get('contents', contents)
            except json.JSONDecodeError:
                pass
        
        results.append((doc_id, score, contents))
        
        if verbose:
            print(f"\n排名 {len(results)}: {doc_id} (分数: {score:.4f})")
            if contents:
                print(f"内容: {contents[:200]}..." if len(contents) > 200 else f"内容: {contents}")
    
    if verbose:
        print(f"\n✅ 检索完成，返回 {len(results)} 个结果")
    
    return results

if __name__ == '__main__':
    import fire
    fire.Fire()