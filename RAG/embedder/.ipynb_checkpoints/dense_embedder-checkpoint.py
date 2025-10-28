# Requires transformers>=4.51.0

import torch

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

max_length = 8192
model_dir = '/home/changc/Bishe/RAG/embedder'

import sys
sys.path.insert(0, model_dir)

import pdb
from .custom_embedder import CustomEmbedder

class DenseEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f'{model_dir}/Qwen3-Embedding-0.6B', padding_side='left')
        self.embedder = CustomEmbedder(f'{model_dir}/Qwen3-Embedding-0.6B', f'{model_dir}/Qwen3-initial_embedder.pth')
        # self.origin_embedder = AutoModel.from_pretrained(f'{model_dir}/Qwen3-Embedding-0.6B', output_hidden_states=True)

    def embedding(self, input):
        batch_dict = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        output = self.embedder(batch_dict)
        def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        return last_token_pool(output.last_hidden_state, batch_dict['attention_mask'])