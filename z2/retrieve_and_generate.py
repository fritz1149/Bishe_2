import torch
import numpy as np
import tempfile
import shutil
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import jieba

#TODO：参数调整&开始跑

@dataclass
class RAGConfig:
    """RAG 系统配置"""
    # 向量检索配置
    vector_index_dir: str  # 流量向量索引目录
    bm25_index_dir: str    # BM25 语料索引目录
    
    # 检索参数
    initial_top_k: int = 10        # 初始检索 top-k
    iterative_top_k: int = 1       # 迭代检索 top-k
    max_iterations: int = 5        # 最大迭代次数
    
    # 结束标志
    stop_phrases: List[str] = None  # 结束性文本列表
    
    def __post_init__(self):
        if self.stop_phrases is None:
            self.stop_phrases = ["结果是", "最终答案", "综上所述", "因此可以判断"]

class RAGRetriever:
    """RAG 检索器，封装向量检索和 BM25 检索"""
    
    def __init__(
        self,
        embedder_args,
        config: RAGConfig,
        device: str = None
    ):
        """
        初始化 RAG 检索器
        
        Args:
            embedder_args: TrafficEmbedder 模型参数
            config: RAG 配置
            device: 设备
        """
        self.embedder_args = embedder_args
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedder = None
        self.vector_index = None
        self.vector_metadata = None
    
    def _load_embedder(self):
        """延迟加载 TrafficEmbedder"""
        if self.embedder is None:
            from z2.model import TrafficEmbedder
            print("⏳ 正在加载 TrafficEmbedder...")
            self.embedder = TrafficEmbedder(self.embedder_args).to(self.device)
            self.embedder.resume(self.embedder_args)
            self.embedder.eval()
            print(f"✅ TrafficEmbedder 已加载 (设备: {self.device})")
    
    def _load_vector_index(self):
        """延迟加载向量索引"""
        if self.vector_index is None:
            from z2.RAG.vector_utils import load_faiss_index
            print(f"⏳ 正在加载向量索引: {self.config.vector_index_dir}")
            self.vector_index, self.vector_metadata = load_faiss_index(self.config.vector_index_dir)
            print(f"✅ 向量索引已加载 (文档数: {self.vector_metadata['num_docs']})")
    
    def unload_embedder(self):
        """卸载 TrafficEmbedder 以释放显存"""
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
            torch.cuda.empty_cache()
            print("🗑️ TrafficEmbedder 已卸载，显存已释放")
    
    def unload_vector_index(self):
        """卸载向量索引以释放内存"""
        if self.vector_index is not None:
            del self.vector_index
            del self.vector_metadata
            self.vector_index = None
            self.vector_metadata = None
            print("🗑️ 向量索引已卸载")
    
    @torch.no_grad()
    def get_traffic_embedding(self, batch_data: Dict) -> np.ndarray:
        """
        获取流量数据的嵌入向量
        
        Args:
            batch_data: collate 函数输出的批次数据字典
        
        Returns:
            numpy 数组形式的嵌入向量
        """
        self._load_embedder()
        # 生成嵌入
        embeddings = self.embedder(**batch_data, normalize=True)
        return embeddings.cpu().numpy()
    
    def search_vector_index(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        在向量索引中检索
        
        Args:
            query_embedding: 查询向量
            k: top-k
        
        Returns:
            [(doc_id, score), ...] 列表
        """
        self._load_vector_index()
        
        assert query_embedding.ndim == 2, f"query_embedding must be 2D, got {query_embedding.ndim}D"
        
        scores, indices = self.vector_index.search(query_embedding, k)
        
        doc_ids = self.vector_metadata['doc_ids']
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                break
            results.append((doc_ids[idx], float(score)))
        
        return results
    
    def search_bm25_by_query(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[str, float, str]]:
        """
        使用文本查询在 BM25 索引中检索
        
        Args:
            query: 查询文本
            k: top-k
        
        Returns:
            [(doc_id, score, contents), ...] 列表
        """
        from z2.RAG.retriever.BM25 import search
        return search(query, self.config.bm25_index_dir, k=k, return_contents=True)
    
    def search_bm25_by_ids(
        self, 
        doc_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据 doc_id 列表在 BM25 索引中查询对应的文本语料
        
        注意: pyserini 的 SimpleSearcher 支持通过 doc_id 直接获取文档
        
        Args:
            doc_ids: 文档 ID 列表
        
        Returns:
            [{'id': str, 'contents': str, ...}, ...] 字典列表
        """
        import json
        from pyserini.search.lucene import LuceneSearcher
        
        searcher = LuceneSearcher(self.config.bm25_index_dir)
        
        results = []
        for doc_id in doc_ids:
            try:
                doc = searcher.doc(doc_id)
                if doc is not None:
                    raw = doc.lucene_document().get('raw')
                    try:
                        doc_dict = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"JSON 解析错误: {raw}")
                        continue
                    results.append(doc_dict)
            except Exception:
                print(f"文档 {doc_id} 获取失败")
                continue
        
        return results


class TempBM25Index:
    """临时 BM25 索引（内存中），用于在初始语料中检索"""
    
    def __init__(self, corpus_list: List[Dict]):
        """
        构建临时 BM25 索引
        
        Args:
            corpus_list: 语料列表，每个元素为 {'id': str, 'contents': str, ...}
        """
        self.corpus_list = corpus_list
        self.doc_ids = [c['id'] for c in corpus_list]
        
        # 分词（简单按空格和标点分割）
        self.tokenized_corpus = []
        for c in corpus_list:
            tokens = self._tokenize(c['contents'])
            self.tokenized_corpus.append(tokens)
        
        # 构建 BM25 索引
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
    
    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 进行中文分词"""
        # 使用 jieba 分词，过滤停用词和空格
        tokens = list(jieba.cut(text))
        # 过滤空字符串和纯空格
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        在临时索引中检索
        
        Args:
            query: 查询文本
            k: top-k
        
        Returns:
            [{'id': str, 'contents': str, 'score': float, ...}, ...] 字典列表
        """
        if self.bm25 is None or not self.corpus_list:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取 top-k 索引
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append({
                    'score': float(scores[idx]),
                    **self.corpus_list[idx]
                })
        
        return results


def get_traffic_corelated_corpus(
    retriever: RAGRetriever,
    batch_data: Dict,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    初始检索：输入流量数据，获取相关的文本语料
    
    流程:
    1. 使用 TrafficEmbedder 生成流量向量
    2. 在流量向量库中检索 top-k 相关向量
    3. 获取这些向量的 id
    4. 在 BM25 语料库中查询 id 对应的文本语料
    
    Args:
        retriever: RAG 检索器实例
        batch_data: collate 函数输出的批次数据 (单个样本时 batch_size=1)
        top_k: 检索数量，默认使用配置中的 initial_top_k
    
    Returns:
        语料列表，每个元素为 {'id': str, 'contents': str, 'score': float}
    """
    if top_k is None:
        top_k = retriever.config.initial_top_k
    
    # 1. 生成流量嵌入向量
    query_embedding = retriever.get_traffic_embedding(batch_data)
    
    # 2. 在向量库中检索 top-k
    vector_results = retriever.search_vector_index(query_embedding, k=top_k)
    
    # 3. 获取 id 列表
    doc_ids = [doc_id for doc_id, _ in vector_results]
    id_to_score = {doc_id: score for doc_id, score in vector_results}
    
    # 4. 在 BM25 语料库中查询对应的文本
    bm25_results = retriever.search_bm25_by_ids(doc_ids)
    
    # 5. 添加 score 字段
    for doc_dict in bm25_results:
        doc_dict['score'] = id_to_score.get(doc_dict['id'], 0.0)
    
    return bm25_results

def retrieve_iteratively(
    generator,  # ProposeModel 实例
    tokenizer,
    batch_data: Dict,
    initial_corpus: List[Dict[str, Any]],
    config: RAGConfig = None
) -> Dict[str, Any]:
    """
    迭代式检索：交替进行推理和检索
    
    输入拼接顺序：语料（前）+ 流量（中）+ 推理（后）
    
    每轮迭代:
    1. 推理环节：基于流量、问题、之前的推理和检索结果，生成一句推理
    2. 检索环节：从整个语料库和临时语料库各检索一条（共2条）
    3. 将结果追加到历史中
    
    终止条件:
    - 迭代次数超过阈值
    - 推理生成了结束性文本（如"结果是"）
    
    Args:
        retriever: RAG 检索器实例
        generator: ProposeModel 实例（用于生成推理）
        tokenizer: tokenizer
        batch_data: collate 函数输出的批次数据（batch_size=1，无PAD填充）
        question: 预置的问题
        initial_corpus: 初始检索得到的语料列表
        config: RAG 配置，默认使用 retriever 的配置
    
    Returns:
        {
            'iterations': List[Dict],  # 每轮迭代的结果
            'all_corpus': List[Dict],  # 所有检索到的语料（去重）
            'reasoning_history': str,  # 完整的推理历史
            'stopped_by': str          # 终止原因: 'max_iterations' 或 'stop_phrase'
        }
    """
    if config is None:
        raise ValueError("config 不能为 None，请传入 RAGConfig 实例")
    
    iterations = []
    all_corpus = {c['id']: c for c in initial_corpus}  # 用 dict 去重
    retrieved_ids = set()  # 已检索到的 ID 集合
    reasoning_history = ""
    stopped_by = "max_iterations"
    
    # 构建临时 BM25 索引（用于在初始语料中检索）
    temp_bm25_index = TempBM25Index(initial_corpus)
    
    device = generator.device if hasattr(generator, 'device') and generator.device else 'cuda'
    
    # 推断 autocast dtype（循环外只检测一次）
    model_dtype = next(generator.parameters()).dtype
    use_autocast = model_dtype in (torch.float16, torch.bfloat16)
    
    # 获取流量数据的各部分
    traffic_input_ids = batch_data['input_ids'].to(device)  # (1, traffic_seq_len)
    traffic_attention_mask = batch_data.get('attention_mask')
    traffic_position_ids = batch_data.get('position_ids')
    traffic_payloads = batch_data.get('payloads')
    
    traffic_seq_len = traffic_input_ids.shape[1]
    
    def _get_first_new_result(results: List[Dict], retrieved_ids: set) -> Optional[Dict]:
        """从检索结果中获取第一个未被添加的结果"""
        for doc_dict in results:
            if doc_dict['id'] not in retrieved_ids and doc_dict['contents']:
                return doc_dict
        return None
    
    for iteration_idx in range(config.max_iterations):
        # 准备当前的语料文本
        corpus_text = "\n".join([
            f"[{i+1}] {c['contents']}" 
            for i, c in enumerate(list(all_corpus.values())[:10])  # 限制语料数量
        ])
        
        # 构建前置 prompt（语料部分）
        system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|>
<|im_start|>user
"""
        corpus_prompt = f"""接下来会给出一些也是针对流量信息的问答语料，这些语料所基于的流量信息将不会被给出，仅有问题和回答会被给出。可以参考其中的推理逻辑或步骤。
相关语料:
{corpus_text}

"""
        # 构建后置 prompt（推理部分）
        if reasoning_history:
            reasoning_prompt = f"""之前的推理:
{reasoning_history}

请继续推理，输出下一步的分析（一句话）。如果已经可以得出结论，请以"结果是"开头给出最终答案。"""
        else:
            reasoning_prompt = """请开始推理，输出第一步的分析（一句话）。"""
        
        # 添加对话结束标记
        reasoning_prompt = reasoning_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码语料部分和推理部分
        # 将 system_prompt 和 corpus_prompt 合并
        full_corpus_prompt = system_prompt + corpus_prompt
        corpus_encoding = tokenizer(full_corpus_prompt, return_tensors='pt', add_special_tokens=True)
        reasoning_encoding = tokenizer(reasoning_prompt, return_tensors='pt', add_special_tokens=False)
        
        corpus_ids = corpus_encoding['input_ids'].to(device)  # (1, corpus_seq_len)
        reasoning_ids = reasoning_encoding['input_ids'].to(device)  # (1, reasoning_seq_len)
        
        corpus_seq_len = corpus_ids.shape[1]
        reasoning_seq_len = reasoning_ids.shape[1]
        
        # 拼接 input_ids: 语料 + 流量 + 推理
        combined_input_ids = torch.cat([corpus_ids, traffic_input_ids, reasoning_ids], dim=1)
        
        # 构建 attention_mask（全为1）
        total_seq_len = corpus_seq_len + traffic_seq_len + reasoning_seq_len
        combined_attention_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=device)
        
        # 构建 position_ids
        # 语料部分: 0, 1, 2, ..., corpus_seq_len-1
        # 流量部分: 在原 position_ids 基础上加上 corpus_seq_len 的偏移
        # 推理部分: corpus_seq_len + traffic_seq_len, ..., total_seq_len-1
        corpus_position_ids = torch.arange(corpus_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(3, -1, -1)
        adjusted_traffic_position_ids = traffic_position_ids.to(device) + corpus_seq_len
        reasoning_position_ids = torch.arange(
            corpus_seq_len + traffic_seq_len, total_seq_len,
            dtype=torch.long, device=device
        ).unsqueeze(0).expand(3, -1, -1)
        
        combined_position_ids = torch.cat([
            corpus_position_ids, 
            adjusted_traffic_position_ids, 
            reasoning_position_ids
        ], dim=2)
        
        # 生成推理
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=model_dtype if use_autocast else None):
            output_ids = generator.generate(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                position_ids=combined_position_ids,
                payloads=traffic_payloads,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        new_reasoning = tokenizer.decode(
            output_ids[0][combined_input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        # full_output = tokenizer.decode(
        #     output_ids[0], 
        #     skip_special_tokens=True
        # ).strip()
        # print("Full output:", full_output)
        
        # 更新推理历史
        if reasoning_history:
            reasoning_history += f"\n{new_reasoning}"
        else:
            reasoning_history = new_reasoning
        
        # 检查是否包含结束性文本
        should_stop = False
        for stop_phrase in config.stop_phrases:
            if stop_phrase in new_reasoning:
                should_stop = True
                stopped_by = f"stop_phrase: {stop_phrase}"
                break
        
        # 2. 检索环节（如果没有结束）
        # 每个检索环节应得出2条结果：1条来自整个语料库，1条来自临时语料库
        new_corpus = []
        if not should_stop:
            # 2.1 从整个 BM25 语料库检索
            from z2.RAG.retriever.BM25 import search as bm25_search
            bm25_results = bm25_search(
                new_reasoning, 
                config.bm25_index_dir,
                k=config.iterative_top_k * 5,  # 多检索一些以便找到未添加的
                return_contents=True
            )
            result_from_bm25 = _get_first_new_result(bm25_results, retrieved_ids)
            if result_from_bm25:
                result_from_bm25['source'] = 'bm25'
                new_corpus.append(result_from_bm25)
                retrieved_ids.add(result_from_bm25['id'])
                all_corpus[result_from_bm25['id']] = result_from_bm25
            
            # 2.2 从临时语料库（初始语料）检索
            temp_results = temp_bm25_index.search(
                new_reasoning, 
                k=config.iterative_top_k * 5
            )
            result_from_temp = _get_first_new_result(temp_results, retrieved_ids)
            if result_from_temp:
                result_from_temp['source'] = 'initial'
                new_corpus.append(result_from_temp)
                retrieved_ids.add(result_from_temp['id'])
                all_corpus[result_from_temp['id']] = result_from_temp
        
        # 记录本轮迭代
        iterations.append({
            'iteration': iteration_idx + 1,
            'reasoning': new_reasoning,
            'new_corpus': new_corpus,
            'corpus_count': len(new_corpus)
        })
        
        if should_stop:
            break
        
        # 清理本轮迭代的中间变量
        del combined_input_ids, combined_attention_mask, combined_position_ids
        del corpus_ids, reasoning_ids, output_ids
        torch.cuda.empty_cache()
    
    return {
        'iterations': iterations,
        'all_corpus': list(all_corpus.values()),
        'reasoning_history': reasoning_history,
        'stopped_by': stopped_by
    }


def generate_response(
    generator,  # ProposeModel 实例
    tokenizer,
    batch_data: Dict,
    corpus_list: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    think_first: bool = True,
    have_corpus: bool = True
) -> str:
    """
    最终生成：组合流量、问题、检索结果，使用 ProposeModel 生成答案
    
    输入拼接顺序：语料（前）+ 流量（中）+ 生成提示（后）
    
    Args:
        generator: ProposeModel 实例
        tokenizer: tokenizer
        batch_data: collate 函数输出的批次数据（batch_size=1，无PAD填充）
        question: 预置的问题
        corpus_list: 检索到的语料列表
        max_new_tokens: 最大生成 token 数
    
    Returns:
        生成的最终答案文本
    """
    device = generator.device if hasattr(generator, 'device') and generator.device else 'cuda'
    
    # 构建语料文本
    corpus_text = "\n".join([
        f"[{i+1}] {c['contents']}" 
        for i, c in enumerate(corpus_list[:10])  # 限制语料数量
    ])
    
    # 构建前置 prompt（语料部分）
    system_prompt = """<|im_start|>system
你是一个AI助手，擅长阅读表格形式的网络流量并对其进行思考和理解，并能够完成各种针对网络流量的问题。<|im_end|>
<|im_start|>user
"""
    corpus_prompt = f"""接下来会给出一些也是针对流量信息的问答语料，这些语料所基于的流量信息将不会被给出，仅有问题和回答会被给出。可以参考其中的推理逻辑或步骤。
相关语料:
{corpus_text}

"""
    # 构建后置 prompt（生成提示部分）
    if think_first:
        generation_prompt = """请注意，给出的流量表格中，ip和端口均经过了随机化处理，因此请不要根据这些字段的取值范围来判断。
请严格按照以下格式输出结果：
推理：[推理过程，不超过300字]
类别：[分类标签]
<|im_end|>
<|im_start|>assistant
"""
    else:
        generation_prompt = """请严格按照以下格式输出结果：
类别：[分类标签]
解释：[解释文本，不超过300字]
<|im_end|>
<|im_start|>assistant
"""

    # 获取流量数据的各部分
    traffic_input_ids = batch_data['input_ids'].to(device)  # (1, traffic_seq_len)
    traffic_attention_mask = batch_data.get('attention_mask')
    traffic_position_ids = batch_data.get('position_ids')
    traffic_payloads = batch_data.get('payloads')
    
    traffic_seq_len = traffic_input_ids.shape[1]
    
    # 编码语料部分和生成提示部分
    # 将 system_prompt 和 corpus_prompt 合并
    full_corpus_prompt = system_prompt + corpus_prompt if have_corpus else system_prompt
    corpus_encoding = tokenizer(full_corpus_prompt, return_tensors='pt', add_special_tokens=True)
    generation_encoding = tokenizer(generation_prompt, return_tensors='pt', add_special_tokens=False)
    
    corpus_ids = corpus_encoding['input_ids'].to(device)  # (1, corpus_seq_len)
    generation_ids = generation_encoding['input_ids'].to(device)  # (1, generation_seq_len)
    
    corpus_seq_len = corpus_ids.shape[1]
    generation_seq_len = generation_ids.shape[1]
    
    # 拼接 input_ids: 语料 + 流量 + 生成提示
    combined_input_ids = torch.cat([corpus_ids, traffic_input_ids, generation_ids], dim=1)
    
    # 构建 attention_mask（全为1）
    total_seq_len = corpus_seq_len + traffic_seq_len + generation_seq_len
    combined_attention_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=device)
    
    # 构建 position_ids
    # 语料部分: 0, 1, 2, ..., corpus_seq_len-1
    # 流量部分: 在原 position_ids 基础上加上 corpus_seq_len 的偏移
    # 生成提示部分: corpus_seq_len + traffic_seq_len, ..., total_seq_len-1
    corpus_position_ids = torch.arange(corpus_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(3, -1, -1)
    adjusted_traffic_position_ids = traffic_position_ids.to(device) + corpus_seq_len
    generation_position_ids = torch.arange(
        corpus_seq_len + traffic_seq_len, total_seq_len,
        dtype=torch.long, device=device
    ).unsqueeze(0).expand(3, -1, -1)
    # print(corpus_position_ids.shape)
    # print(adjusted_traffic_position_ids.shape)
    # print(generation_position_ids.shape)
    
    combined_position_ids = torch.cat([
        corpus_position_ids, 
        adjusted_traffic_position_ids, 
        generation_position_ids
    ], dim=2)
    
    # 生成
    # 根据模型参数自动推断 autocast dtype
    model_dtype = next(generator.parameters()).dtype
    use_autocast = model_dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_autocast, dtype=model_dtype if use_autocast else None):
        output_ids = generator.generate(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            position_ids=combined_position_ids,
            payloads=traffic_payloads,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # 解码生成的文本（只取新生成的部分）
    response = tokenizer.decode(
        output_ids[0][combined_input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # return response.strip()

    original = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    
    # 清理中间变量释放显存
    del combined_input_ids, combined_attention_mask, combined_position_ids
    del corpus_ids, generation_ids, output_ids
    torch.cuda.empty_cache()
    
    return response.strip(), original.strip()


def run_rag_pipeline(
    retriever: RAGRetriever,
    generator,  # ProposeModel 实例
    tokenizer,
    batch_data: Dict,
    question: str,
    enable_iterative: bool = True,
    max_new_tokens: int = 512,
    unload_embedder_after_initial: bool = True
) -> Dict[str, Any]:
    """
    运行完整的 RAG 流程
    
    Args:
        retriever: RAG 检索器实例
        generator: ProposeModel 实例
        tokenizer: tokenizer
        batch_data: collate 函数输出的批次数据
        question: 问题
        enable_iterative: 是否启用迭代式检索
        max_new_tokens: 最大生成 token 数
        unload_embedder_after_initial: 初始检索完成后是否卸载 embedder 以节省显存
    
    Returns:
        {
            'initial_corpus': List[Dict],    # 初始检索结果
            'iterative_result': Dict,        # 迭代检索结果（如果启用）
            'final_corpus': List[Dict],      # 最终使用的语料
            'response': str                  # 最终生成的答案
        }
    """
    # 1. 初始检索
    print("🔍 执行初始检索...")
    initial_corpus = get_traffic_corelated_corpus(retriever, batch_data)
    print(f"   - 检索到 {len(initial_corpus)} 个相关语料")
    
    # 初始检索完成后卸载 embedder 以节省显存
    if unload_embedder_after_initial:
        retriever.unload_embedder()
        retriever.unload_vector_index()
    
    # 2. 迭代式检索（可选）
    iterative_result = None
    final_corpus = initial_corpus
    reasoning_history = None
    
    if enable_iterative:
        print("🔄 执行迭代式检索...")
        iterative_result = retrieve_iteratively(
            retriever=retriever,
            generator=generator,
            tokenizer=tokenizer,
            batch_data=batch_data,
            question=question,
            initial_corpus=initial_corpus
        )
        final_corpus = iterative_result['all_corpus']
        reasoning_history = iterative_result['reasoning_history']
        print(f"   - 迭代 {len(iterative_result['iterations'])} 轮")
        print(f"   - 最终语料数: {len(final_corpus)}")
        print(f"   - 终止原因: {iterative_result['stopped_by']}")
    
    # 3. 最终生成
    print("✍️ 生成最终答案...")
    response = generate_response(
        generator=generator,
        tokenizer=tokenizer,
        batch_data=batch_data,
        question=question,
        corpus_list=final_corpus,
        max_new_tokens=max_new_tokens
    )
    
    print("✅ RAG 流程完成")
    
    return {
        'initial_corpus': initial_corpus,
        'iterative_result': iterative_result,
        'final_corpus': final_corpus,
        'response': response
    }

if __name__ == "__main__":
    import fire
    fire.Fire()
