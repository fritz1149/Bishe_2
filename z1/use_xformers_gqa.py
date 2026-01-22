"""
使用 xFormers 的 memory_efficient_attention 来支持 GQA
通过 Hugging Face 的 AttentionInterface 注册自定义 attention 函数
参考: https://hf-cdn.sufy.com/docs/transformers/v4.57.1/en/attention_interface
"""
import torch
from typing import Optional
from transformers import AttentionInterface

try:
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    xops = None
    print("警告: xformers 未安装，请运行: pip install xformers")

def xformers_gqa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    使用 xFormers 的 memory_efficient_attention 实现，支持 GQA
    
    重要说明：
    1. 此函数是一个纯计算单元，不包含任何可训练参数！
    2. 所有模型参数（q_proj, k_proj, v_proj, o_proj 等）都在模型的 attention 层中
    3. 此函数只负责执行 attention 的数学计算，不涉及参数的参与
    
    参数使用流程：
    1. 在调用此函数之前，模型的 q_proj, k_proj, v_proj 已经计算了 Q、K、V
    2. 此函数接收的是已经通过模型参数计算好的 query, key, value
    3. 此函数只负责用 xFormers 的方式计算 attention（纯计算，无参数）
    4. 返回后，模型的 o_proj 会对结果进行输出投影
    
    参数:
        module: attention 模块（包含 num_key_value_groups 等属性和所有参数）
        query: [B, num_attention_heads, M, head_dim] - 已通过 q_proj 和 q_norm 计算
        key: [B, num_key_value_heads, M, head_dim] - 已通过 k_proj 和 k_norm 计算
        value: [B, num_key_value_heads, M, head_dim] - 已通过 v_proj 计算
        attention_mask: 可选的 attention mask
        dropout: dropout 概率
        scaling: 缩放因子（如果为 None，使用默认的 1/sqrt(head_dim)）
        is_causal: 是否为因果 attention
        **kwargs: 其他参数
    
    返回:
        (attn_output, attn_weights): attn_output 为 [B, num_attention_heads, M, head_dim]
        注意：返回的 attn_output 会被模型的 o_proj 进一步处理
    """
    if not XFORMERS_AVAILABLE:
        raise ImportError("xformers 未安装，请运行: pip install xformers")
    # print("query shape:", query.shape)
    # print("key shape:", key.shape)
    # print("value shape:", value.shape)
    # 获取维度信息
    # query 的形状是 [B, H, M, K]，来源于 Qwen3VL 模型代码：
    # - modeling_qwen3_vl.py:429: query_states = ...transpose(1, 2)
    # - 将 [B, M, H, K] 转换为 [B, H, M, K] 格式
    # - 这是所有 attention_interface 函数接收的标准格式
    # 参考：transformers/integrations/sdpa_attention.py 中的注释和实现
    batch_size, num_heads_q, seq_len_q, head_dim = query.shape  # [B, H, M, K]
    seq_len_kv = key.shape[2]
    num_kv_heads = key.shape[1]  # key 和 value 是 [B, num_kv_heads, M, K]
    
    # 处理 GQA 情况
    assert seq_len_kv == value.shape[2]
    assert num_heads_q != num_kv_heads
    assert num_heads_q % num_kv_heads == 0
    # GQA: 需要 reshape 为 xFormers 的 [B, M, G, H, K] 格式
    # 根据 xFormers 文档: https://facebookresearch.github.io/xformers/components/ops.html
    num_groups = num_kv_heads  # G = num_key_value_heads
    heads_per_group = num_heads_q // num_groups  # H = num_attention_heads // num_key_value_heads

    
    # Reshape Q: [B, H, M, K] -> [B, M, H, K] -> [B, M, G, H_per_group, K]
    query = query.transpose(1, 2)  # [B, M, H, K]
    query = query.view(batch_size, seq_len_q, num_groups, heads_per_group, head_dim)
    
    # Reshape K, V: [B, num_kv_heads, M, K] -> [B, M, num_kv_heads, K] -> [B, M, G, 1, K] -> expand to [B, M, G, H_per_group, K]
    key = key.transpose(1, 2)  # [B, M, num_kv_heads, K]
    key = key.view(batch_size, seq_len_kv, num_groups, 1, head_dim)
    key = key.expand(batch_size, seq_len_kv, num_groups, heads_per_group, head_dim)
    
    value = value.transpose(1, 2)  # [B, M, num_kv_heads, K]
    value = value.view(batch_size, seq_len_kv, num_groups, 1, head_dim)
    value = value.expand(batch_size, seq_len_kv, num_groups, heads_per_group, head_dim)
    
    # 处理 attention_mask（如果需要）
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    
    # 使用 xFormers memory_efficient_attention
    # 注意：xFormers 的 GQA 支持是实验性的，仅支持 forward pass
    attn_output = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attention_mask,
        p=dropout if module.training else 0.0,
        scale=scaling,
    )
    
    # Reshape 回原始格式: [B, M, G, H_per_group, K] -> [B, M, H, K] -> [B, H, M, K]
    attn_output = attn_output.reshape(batch_size, seq_len_q, num_heads_q, head_dim)
    # 转回 [B, H, M, K] 格式，与 transformers 接口约定一致
    # 参考：sdpa_attention.py:106 也有类似的 transpose(1, 2)
    attn_output = attn_output.transpose(1, 2)  # [B, M, H, K] → [B, H, M, K]
    return attn_output, None

if __name__ == "__main__":
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    # 确保已注册
    if not XFORMERS_AVAILABLE:
        print("请先安装 xformers: pip install xformers")
        exit(1)
    
    # 加载模型时指定使用 xformers_gqa
    print("加载模型（使用 xformers_gqa attention）...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "./Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="xformers_gqa"  # 使用注册的自定义 attention
    )
    
    # 或者加载后动态切换
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "./Qwen3-VL-8B-Instruct",
    #     trust_remote_code=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    # model = setup_xformers_attention(model)
    
    # 测试
    processor = AutoProcessor.from_pretrained("./Qwen3-VL-8B-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello, nice to meet you."},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    print("使用 xFormers GQA 进行推理...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    print("完成！")
    print(f"生成结果: {processor.decode(generated_ids[0], skip_special_tokens=True)}")
