import torch
from transformers import AutoModel

def initial_embedding():
    model = AutoModel.from_pretrained('./Qwen3-Embedding-0.6B')
    embed_tokens = model.embed_tokens
    
    # # 保存新模型及其参数
    torch.save(embed_tokens.state_dict(), 'Qwen3-initial_embedder.pth')

if __name__ == '__main__':
    initial_embedding()