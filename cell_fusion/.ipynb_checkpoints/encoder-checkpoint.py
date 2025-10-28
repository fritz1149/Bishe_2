import torch.nn as nn
import torch
from typing import Optional
from transformers import AutoModel, AutoTokenizer

seq_len = 40
d_model = 2560
nhead = 8  # 自注意力机制的头数
num_layers = 1  # 编码器层的数量
dim_feedfoward = d_model * 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomRotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.precompute_freqs_cis(dim, seq_len)

    def precompute_freqs_cis(self, dim: int, seq_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.freqs_cis = self.freqs_cis.view(1, *self.freqs_cis.shape).to(device)
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_ = torch.view_as_complex(x_)
        x_out = torch.view_as_real(x_ * self.freqs_cis[:,:x_.shape[1],:]).flatten(2)
        return x_out.type_as(x)

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rotary_pos_emb = CustomRotaryEmbedding(d_model, seq_len)

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.rotary_pos_emb(x)
        return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)
        
class CustomEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedfoward,  # 前馈神经网络中的隐藏层维度
            dropout=0.1,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_len * d_model, d_model)

    def forward(self, embeddings):
        hidden_states = self.encoder(embeddings)
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        paddings = torch.zeros(hidden_states.shape[0], seq_len * d_model - hidden_states.shape[1]).to(device)
        hidden_states = torch.cat((hidden_states, paddings), 1)
        hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states.view(hidden_states.shape[0], 1, hidden_states.shape[1])
        return hidden_states

if __name__ == '__main__':
    cell_encoder = CustomEncoder('Qwen3-4B')
    tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B', padding_side='left')
    body_llm = AutoModel.from_pretrained('Qwen3-4B')
    embed_tokens = body_llm.embed_tokens
    
    text = '192.168.137.123'
    batch_dict = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length = 8192,
            return_tensors="pt",
        )
    embeddings = embed_tokens(batch_dict["input_ids"])
    
    output = cell_encoder(embeddings)
    print(output.shape)