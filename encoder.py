import torch
import torch.nn as nn

class Longformer(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.fc = nn.Identity()

    #TODO：输入格式
    def forward(self, x):
        input_ids, attention_mask, global_attention_mask = x
        # CLS
        x = self.original_model(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x


# TODO：分成三种，初始化、加载模型、空模型。初始化的要从longformer借position_embeddings，空模型仅是结构、为了加载模型做铺垫
def get_longformer(args):
    # 获取tokenizer
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from preprocess.utils import _Qwen3VL_tokenizer
    tokenizer, _, _, _, _, _ = _Qwen3VL_tokenizer()

    # 获取模型
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained("./longformer-base-4096")
    new_model = AutoModel.from_config(config)

    # 获取position_embeddings
    model = AutoModel.from_pretrained("./longformer-base-4096")
    with torch.no_grad():
        # 旧表 shape: [max_pos, hidden]
        old_pos_emb = model.embeddings.position_embeddings.weight
        # 直接赋值给新模型
        new_model.embeddings.position_embeddings.weight.copy_(old_pos_emb)

    hidden = config.hidden_size  # 768
    # 新建一个更小词表的 embedding 层
    new_token_emb = nn.Embedding(len(tokenizer.vocab), hidden)
    nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_model.embeddings.word_embeddings = new_token_emb

    return Longformer(new_model), hidden

def get_longformer_with_projector(args):
    model, hidden_size = get_longformer(args)
    # TODO：projector的输出维度适配backbone LLM
    if args.projector == 'mlp':
        from simsiam import mlp_mapper
        projector = mlp_mapper(hidden_size, args.projector_arch, bn_end=False)
    elif args.projector == 'linear':
        projector = nn.Linear(hidden_size, args.linear_output_dim)
    else:
        raise ValueError(f"Unknown projector: {args.projector}")
    model.fc = projector
    return model

def transform_headers(args):
    pass

if __name__ == '__main__':
    from uer.uer.opts import tokenizer_opts
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tokenizer_opts(parser)
    args = parser.parse_args()
    args.tokenizer = 'bert'
    args.vocab_path = 'config/encryptd_vocab.txt'

    import time
    start_time = time.time()
    model, hidden_size = get_longformer(args)
    end_time = time.time()
    print(f"get_longformer 用时: {end_time - start_time:.4f} 秒")
    print(model)
    print(hidden_size)