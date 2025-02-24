import torch
from torch import nn
import math

def encode_mask(seq_len, valid_lens, device:None):
    """
    计算编码器掩码
    Args:
        seq_len (int): 序列长度
        valid_lens (Tensor): shape [batch_size] or [batch_size, seq_len] 有效长度
        device(str): 在device运行存储张量

    Returns:
        Tensor: dtype bool, shape [batch_size, seq_len, seq_len], 编码器掩码
    """
    if valid_lens is None:
        return None
    elif valid_lens.dim() == 1:
        valid_lens = valid_lens.repeat_interleave(seq_len).reshape(-1, seq_len)
    else:
        pass
    
    return torch.arange(seq_len, device=device).repeat(valid_lens.shape[0], 1).unsqueeze(1) >= valid_lens.unsqueeze(2)
    # mask = torch.arange(seq_len, device=device).unsqueeze(0) >= valid_lens.unsqueeze(1)
    # return mask.unsqueeze(1).repeat(1, seq_len, 1)


class DotPructAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def forward(self, queries, keys, values, valid_lens, device: None):
        """计算缩放点积注意力

        Args:
            queries (Tensor): shape [batch_size * num_heads, seq_len, h_d_model]
            keys (Tensor): shape [batch_size * num_heads, seq_len, h_d_model]
            values (Tensor): shape [batch_size * num_heads, seq_len, h_d_model]
            valid_lens (Tensor): shape [batch_size] or [batch_size, seq_len] 有效长度
            device (str): 在device运行存储张量

        Returns:
            Tensor: shape [batch_size * num_heads, seq_len, h_d_model]， 注意力矩阵
        """
        # 向量维度
        d = queries.shape[-1]
        # 缩放点积
        attention_score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 计算编码器掩码，遮挡注意力得分
        seq_len = queries.shape[1]
        mask = encode_mask(seq_len, valid_lens, device)
        mask = mask.repeat_interleave(queries.shape[0] // valid_lens.shape[0], dim=0)
        attention_score = attention_score.masked_fill(mask, 1e-6)
        # 对注意力得分softmax，然后和values做矩阵乘法
        return torch.bmm(torch.softmax(attention_score, dim=2), values)


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = DotPructAttention(*args, **kwargs)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    

    def forward(self, queries, keys, values, valid_lens, device: None):
        """
        计算多头注意力
        Args:
            queries (Tensor): shape [batch_size, seq_len, d_model]
            keys (Tensor): shape [batch_size, seq_len, d_model]
            values (Tensor): shape [batch_size, seq_len, d_model]
            valid_lens (Tensor): shape [batch_size] or [batch_size, seq_len] 有效长度
            device (str): 在device运行存储张量

        Returns:
            Tensor: shape [batch_size, seq_len, d_model]
        """
        h_d_model = self.d_model // self.num_heads

        q = self.w_q(queries)
        k = self.w_k(keys)
        v = self.w_v(values)

        batch_size = queries.shape[0]   # 小批量大小
        seq_len = queries.shape[1]  # 序列长度
        q = q.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)   # shape [batch_size, num_heads, seq_len, h_d_model]
        k = k.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, h_d_model).transpose(1, 2)

        # q,k,v变形为[batch_size * num_heads, seq_len, h_d_model]
        q = q.reshape(batch_size * self.num_heads, seq_len, h_d_model)
        k = k.reshape(batch_size * self.num_heads, seq_len, h_d_model)
        v = v.reshape(batch_size * self.num_heads, seq_len, h_d_model)

        # 计算注意力
        attention_output = self.attention(q, k, v, valid_lens, device)
        # 拼接多头注意力
        attention_output = attention_output.reshape(batch_size, self.num_heads, seq_len, h_d_model).transpose(1,2).reshape(batch_size, seq_len, -1)
        # 多头注意力输出
        return self.w_o(attention_output)



        