"""
文本语料筛选模块

主要功能：
使用 AutoModelForCausalLM (Qwen3) 对生成的文本语料进行质量筛选
筛选标准：语言自然度、条理性、问答对应度
"""

import json
import os
import re
import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class CorpusFilter:
    """
    文本语料筛选器 - 使用 LLM 对语料进行质量评估和筛选
    """
    
    def __init__(
        self,
        model_path: str = "Qwen3-1.7B",
        device: str = None,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        初始化语料筛选器
        
        Args:
            model_path: Qwen3 模型路径
            device: 设备 ('cuda' 或 'cpu')，None 则自动选择
            torch_dtype: 模型数据类型
        """
        self.model_path = model_path
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            print(f"📦 正在加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            self.model.eval()
            print(f"✅ 模型已加载 (设备: {self.device})")
    
    def _build_scoring_prompt(self, data: List) -> str:
        """
        构建评分 prompt
        
        Args:
            data: 同一 id 下的所有语料内容列表
            
        Returns:
            评分 prompt 字符串
        """
        contents_text = "\n\n".join([
            f"【语料 {i+1}】\n{content}" for i, content in enumerate(data['contents'])
        ])
        
        prompt = f"""<|im_start|>system
你是一个专业的文本质量评估专家，擅长评估文本的语言质量和内容质量。<|im_end|>
<|im_start|>user
请仔细阅读接下来给出的针对对网络流量的问题和回答，然后从以下三个方面为回答部分打分：

1. 语言自然度（1-10分）：评估文本是否流畅自然，像母语者撰写的中文一样。考虑语法正确性、词汇使用是否地道、句子是否连贯、无尴尬表达。
2. 条理性（1-10分）：评估文本的逻辑结构是否清晰、条理分明。考虑内容是否层层递进、无跳跃、要点是否组织良好、是否有清晰的开头/结尾。
3. 问答对应度（1-10分）：评估生成的答案是否直接、完整地对应并回应了用户的问题。考虑是否切中问题核心、是否完整覆盖问题要点、是否有答非所问、跑题、遗漏关键部分或包含大量无关冗余内容。

以下是注意事项：

1. 网络流量本身的信息将不会被给出，请仅根据问答进行打分。
2. 不要重复给定的问题和回答。
3. 只输出评分结果，不要输出其他解释。
4. 分数必须是1-10之间的整数。
5. 必须对所有语料都进行评分。
6. 语言自然度低于5分的语料将被排除，其他语料则按语言自然度、条理性、问答对应度的顺序排序，取排名第一的语料作为该 id 的唯一留存语料。

问题：
{data['question']}
待评分语料：
{contents_text}

...

请严格按照以下格式输出每条语料的评分，每行一条：
语料1: 语言自然度=X, 条理性=Y, 问答对应度=Z。
语料2: 语言自然度=X, 条理性=Y, 问答对应度=Z。
/no_think
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _parse_scores(self, score_text: str, num_contents: int) -> List[Dict[str, int]]:
        """
        解析 LLM 输出的评分结果
        
        Args:
            score_text: LLM 输出的评分文本
            num_contents: 语料数量
            
        Returns:
            评分列表，每个元素是包含三个评分的字典
            
        Raises:
            ValueError: 解析失败
        """
        scores = []
        
        # 匹配 "语料X: 语言自然度=A, 条理性=B, 问答对应度=C" 格式
        pattern = r'语料\s*(\d+)\s*[:：]\s*语言自然度\s*[=＝]\s*(\d+)\s*[,，]\s*条理性\s*[=＝]\s*(\d+)\s*[,，]\s*问答对应度\s*[=＝]\s*(\d+)'
        matches = re.findall(pattern, score_text)
        
        if len(matches) < num_contents:
            raise ValueError(f"评分解析失败: 期望 {num_contents} 条评分，实际解析到 {len(matches)} 条")
        
        # 按语料编号排序
        matches_sorted = sorted(matches, key=lambda x: int(x[0]))
        
        for match in matches_sorted[:num_contents]:
            idx, naturalness, coherence, relevance = match
            scores.append({
                'naturalness': int(naturalness),
                'coherence': int(coherence),
                'relevance': int(relevance)
            })
        
        return scores
    
    @torch.no_grad()
    def filter_corpus(
        self,
        input_dir: str,
        output_path: str,
        naturalness_threshold: int = 5,
        max_new_tokens: int = 256,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        对语料进行筛选
        
        筛选逻辑：
        1. 读取每个 id 对应的所有语料
        2. 使用 LLM 对每条语料评分（语言自然度、条理性、问答对应度）
        3. 排除语言自然度低于阈值的语料
        4. 对剩余语料按优先级排序（语言自然度 > 条理性 > 问答对应度）
        5. 取第一位作为该 id 的唯一留存语料
        
        Args:
            input_dir: 输入语料目录（包含 .jsonl 文件）
            output_path: 输出文件路径（.jsonl）
            naturalness_threshold: 语言自然度阈值，低于此值的语料被排除
            max_new_tokens: 最大生成 token 数
            verbose: 是否打印详细信息
            
        Returns:
            统计信息字典
        """
        self._load_model()
        
        # 读取所有语料（每条记录包含 id, question, contents）
        print(f"📂 正在读取语料目录: {input_dir}")
        corpus_data = []
        
        jsonl_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jsonl')])
        if not jsonl_files:
            raise FileNotFoundError(f"目录下没有找到 .jsonl 文件: {input_dir}")
        
        for jsonl_file in jsonl_files:
            file_path = os.path.join(input_dir, jsonl_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    # 确保 contents 是列表
                    if isinstance(entry['contents'], str):
                        entry['contents'] = [entry['contents']]
                    corpus_data.append(entry)
        
        total_ids = len(corpus_data)
        print(f"   - 总 ID 数量: {total_ids}")
        print(f"   - 语言自然度阈值: {naturalness_threshold}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 筛选统计
        retained_count = 0
        discarded_count = 0
        error_count = 0
        retained_corpus = []
        
        # 逐条进行筛选
        for data in tqdm(corpus_data, desc="筛选语料"):
            sample_id = data['id']
            contents = data['contents']
            
            if len(contents) == 0:
                discarded_count += 1
                continue
            
            try:
                # 构建评分 prompt
                prompt = self._build_scoring_prompt(data)
                inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成评分
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                score_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                if verbose:
                    origin_text = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    print(f"📝 ID {sample_id} 评分结果: {origin_text}...")
                
                # 解析评分
                scores = self._parse_scores(score_text, len(contents))
                
                # 筛选：排除语言自然度低于阈值的
                valid_indices = [
                    i for i, s in enumerate(scores) 
                    if s['naturalness'] >= naturalness_threshold
                ]
                
                if not valid_indices:
                    # 所有语料都被排除
                    discarded_count += 1
                    continue
                
                # 排序：按 (语言自然度, 条理性, 问答对应度) 降序
                valid_indices.sort(
                    key=lambda i: (
                        scores[i]['naturalness'],
                        scores[i]['coherence'],
                        scores[i]['relevance']
                    ),
                    reverse=True
                )
                
                # 取第一位
                best_idx = valid_indices[0]
                retained_corpus.append({
                    'id': sample_id,
                    'question': data.get('question', ''),
                    'contents': f'问题：{data.get("question", "")}\n回答：{contents[best_idx]}',
                    'scores': scores[best_idx]
                })
                retained_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ ID {sample_id} 处理失败: {str(e)}")
                error_count += 1
                discarded_count += 1
        
        # 保存结果
        filtered_output_path = os.path.join(output_path, 'filtered.jsonl')
        os.makedirs(output_path, exist_ok=True)
        with open(filtered_output_path, 'w', encoding='utf-8') as f:
            for entry in retained_corpus:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 统计信息
        retention_rate = retained_count / total_ids if total_ids > 0 else 0
        
        print(f"\n✅ 语料筛选完成！")
        print(f"   - 总 ID 数量: {total_ids}")
        print(f"   - 有留存语料的 ID: {retained_count}")
        print(f"   - 无留存语料的 ID: {discarded_count}")
        print(f"   - 处理错误数量: {error_count}")
        print(f"   - 留存比例: {retention_rate:.2%}")
        print(f"   - 输出文件: {output_path}")
        
        return {
            'total_ids': total_ids,
            'retained_count': retained_count,
            'discarded_count': discarded_count,
            'error_count': error_count,
            'retention_rate': retention_rate
        }


def run_corpus_filter_pipeline(
    # 输入输出路径
    input_dir: str,
    output_path: str,
    # 模型参数
    model_path: str = "Qwen3-1.7B",
    device: str = None,
    # 筛选参数
    naturalness_threshold: int = 5,
    max_new_tokens: int = 256,
    verbose: bool = True
):
    """
    文本语料筛选流水线
    
    Args:
        input_dir: 输入语料目录（包含 skip_clustering=True 生成的 .jsonl 文件）
        output_path: 输出文件路径（.jsonl）
        model_path: Qwen3 模型路径
        device: 设备
        naturalness_threshold: 语言自然度阈值（1-10），低于此值的语料被排除
        max_new_tokens: 最大生成 token 数
        verbose: 是否打印详细信息
    """
    print("=" * 60)
    print("文本语料筛选流水线")
    print("=" * 60)
    
    # 创建筛选器
    corpus_filter = CorpusFilter(
        model_path=model_path,
        device=device
    )
    
    # 执行筛选
    stats = corpus_filter.filter_corpus(
        input_dir=input_dir,
        output_path=output_path,
        naturalness_threshold=naturalness_threshold,
        max_new_tokens=max_new_tokens,
        verbose=verbose
    )
    
    print("\n" + "=" * 60)
    print("✅ 文本语料筛选流水线完成！")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    import fire
    fire.Fire()
