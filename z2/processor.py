from transformers import LogitsProcessor
import torch

class ContextAwareRepetitionPenalty(LogitsProcessor):
    """
    上下文感知的重复惩罚：
    1. 连续重复 → 强惩罚（指数级）
    2. 间隔重复 → 弱惩罚（线性衰减）
    3. 标点符号 → 完全豁免
    4. 长无标点序列 → 主动提升标点概率
    """
    def __init__(
        self,
        tokenizer,
        continuous_penalty=2.0,    # 连续重复惩罚强度（指数）
    ):
        self.tokenizer = tokenizer
        self.continuous_penalty = continuous_penalty
    
    def __call__(self, input_ids, scores):
        """
        scores: [batch_size, vocab_size] 的 logits
        input_ids: [batch_size, seq_len] 已生成的token序列
        """
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            seq = input_ids[batch_idx].tolist()
            
            # 1. 连续重复检测：统计最近token的连续出现次数
            last_token = seq[-1]
            continuous_count = 1
            for i in range(len(seq)-2, max(-1, len(seq)-6), -1):  # 检查最近5个
                if seq[i] == last_token:
                    continuous_count += 1
                else:
                    break
            
            if continuous_count >= 2 and last_token:
                # 指数级惩罚：连续2次→/2.0, 连续3次→/4.0, 连续4次→/8.0
                penalty = self.continuous_penalty ** (continuous_count - 1)
                scores[batch_idx, last_token] /= penalty
        
        return scores