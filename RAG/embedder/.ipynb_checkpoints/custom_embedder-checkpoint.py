import torch.nn as nn
import torch

from transformers import AutoModel

class CustomEmbedder(nn.Module):
    def __init__(self, qwen3_embedding_path, embed_tokens_path):
        super().__init__()
        self.qwen3_embedding =  AutoModel.from_pretrained(qwen3_embedding_path)
        config = self.qwen3_embedding.config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.qwen3_embedding.padding_idx)
        self.embed_tokens.load_state_dict(torch.load(embed_tokens_path))

    def forward(self, batch_dict):
        embeddings = self.embed_tokens(batch_dict["input_ids"])
        return self.qwen3_embedding.forward(attention_mask=batch_dict['attention_mask'],
                                           inputs_embeds=embeddings)
        