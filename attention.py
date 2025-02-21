import torch
from torch import nn

# 多头注意力

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def forward(self, queries, keys, values, valid_lens):
        pass