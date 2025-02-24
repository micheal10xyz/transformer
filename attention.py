import torch
from torch import nn
import math


class DotPructAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def forward(self, queries, keys, values, valid_lens):
       # 向量维度
       d = queries.shape[-1]
       # 缩放点积
       attention_score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
       attention_score.masked_fill


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = None
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    

    def forward(self, queries, keys, values, valid_lens):
        h_d_model = self.d_model / self.num_heads

        q = torch.bmm(queries, self.w_q)
        k = torch.bmm(keys, self.w_k)
        v = torch.bmm(values, self.w_v)

        batch_size = queries.shape[0]   # 小批量大小
        seq_len = queries.shape[1]  # 序列长度
        q = q.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)   # shape [batch_size, num_heads, seq_len, h_d_model]
        k = k.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)

        # q,k,v变形为[batch_size * num_heads, seq_len, h_d_model]
        q = q.reshape(batch_size * seq_len, self.num_heads, h_d_model)
        k = k.reshape(batch_size * seq_len, self.num_heads, h_d_model)
        v = v.reshape(batch_size * seq_len, self.num_heads, h_d_model)

        attention_score = self.attention(q, k, v, valid_lens)
        # 输出矩阵
        pass