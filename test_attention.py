import torch
from torch import nn
import attention


def test_multi_head_attention_01():
    d = 8
    my_attention = attention.MultiHeadAttention(d, 2)
    nn_attention = nn.MultiheadAttention(d, 2)
    my_attention.w_q.weight.data = nn_attention.in_proj_weight[:d, :]
    my_attention.w_q.bias.data = nn_attention.in_proj_bias[:d]
    my_attention.w_k.weight.data = nn_attention.in_proj_weight[d:2*d, :]
    my_attention.w_k.bias.data = nn_attention.in_proj_bias[d:2*d]
    my_attention.w_v.weight.data = nn_attention.in_proj_weight[2*d:, :]
    my_attention.w_v.bias.data = nn_attention.in_proj_bias[2*d:]
    my_attention.w_o.weight.data = nn_attention.out_proj.weight.data
    my_attention.w_o.bias.data = nn_attention.out_proj.bias.data

    queries = torch.arange(48, dtype=torch.float).reshape(2, 3, 8)
    keys = values = torch.ones(2, 2, 8, dtype=torch.float)
    print(queries)

    valid_lens=torch.tensor([1, 2], dtype=torch.int)

    my_score = my_attention(queries, keys, values, valid_lens)
    nn_score, _ = nn_attention(queries.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1))
    nn_score = nn_score.transpose(0, 1)
    print(my_score)
    print(nn_score)

    mask = attention.encode_mask(3, 2, valid_lens)

    
    assert torch.allclose(my_score, nn_score)
