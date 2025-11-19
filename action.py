# INSERT_YOUR_CODE

import os

def register_special_tokens(tokenizer, token_infos):
    """
    向给定的tokenizer注册特殊token。

    Args:
        tokenizer: Huggingface的Tokenizer对象
        token_infos: List[Tuple[str, str]]，每项为(token类别, token文字串)。如:
            [
              ('pad_token', '<|endoftext|>'),
              ('eos_token', '<|im_end|>'),
              ('additional_special_tokens', '<|object_ref_start|>'),
              ...
            ]
    """
    special_tokens_dict = {}
    # 分类整合
    add_specials = []
    for cat, val in token_infos:
        if cat == "pad_token":
            special_tokens_dict['pad_token'] = val
        elif cat == "eos_token":
            special_tokens_dict['eos_token'] = val
        elif cat == "bos_token":
            special_tokens_dict['bos_token'] = val
        elif cat == "unk_token":
            special_tokens_dict['unk_token'] = val
        elif cat == "additional_special_tokens":
            add_specials.append(val)
    if add_specials:
        special_tokens_dict["additional_special_tokens"] = add_specials

    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def register_dir_special_tokens(tokenizer_src_dir, token_infos, tokenizer_dest_dir):
    """
    加载tokenizer并注册特殊token，然后保存到指定目录。

    Args:
        tokenizer_src_dir: Tokenizer来源目录(str)
        token_infos: List[Tuple[str, str]]，见register_special_tokens
        tokenizer_dest_dir: 注册完special tokens后保存的目标目录(str)
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src_dir)
    tokenizer = register_special_tokens(tokenizer, token_infos)
    os.makedirs(tokenizer_dest_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dest_dir)

# 用法示例:
if __name__ == "__main__":
    # 假定special_token_id_map.txt同目录下
    special_token_path = os.path.join(os.path.dirname(__file__), "special_token_id_map.txt")
    token_infos = []
    with open(special_token_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            tcat, rem = line.split(":", 1)
            token_str = rem.split("->")[0].strip().strip("'\"")
            token_infos.append((tcat.strip(), token_str))
    # 指定huggingface tokenizer原始目录和保存目录
    tokenizer_src_dir = "/path/to/your/tokenizer"
    tokenizer_dest_dir = "/path/to/saved/tokenizer/with_specials"
    register_dir_special_tokens(tokenizer_src_dir, token_infos, tokenizer_dest_dir)

