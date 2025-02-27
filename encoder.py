import torch
from torch import nn
import positional_encoding
import attention
import ffn
import math

class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_ffn_hiddens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.position_wise_ffn = ffn.PositionWiseFFN(d_model, num_ffn_hiddens, d_model)


    def forward(self, blk_in, valid_lens):
        """编码器块

        Args:
            blk_in (Tensor): shape [batch_size, seq_len, d_model]
            valid_lens (Tensor): shape [batch_size]
        """
        # 计算点积注意力
        attention = self.attention(blk_in, blk_in, blk_in, valid_lens)
        # 残差连接，层归一化
        ffn_in = self.norm1(blk_in + attention)
        # 多层感知机(mlp)
        ffn_out = self.position_wise_ffn(ffn_in)
        # 残差连接，层归一化
        return self.norm2(ffn_in + ffn_out)


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block_' + str(i), Block(d_model, num_heads, num_ffn_hiddens))

    
    def forward(self, enc_in, valid_lens):
        """对输入编码
        Args:
            enc_in (Tensor): shape [batch_size, seq_len]，编码器输入
            valid_lens (Tensor): shape [batch_size]，有效长度
        """
        # 词嵌入（embedding）
        embeddings = self.embedding(enc_in) * math.sqrt(self.d_model)
        # 添加位置编码
        blk_in = self.positional_encoding(embeddings)
        # 循环执行编码器块
        for blk in self.blks:
            blk_in = blk(blk_in, valid_lens)
        # 返回编码器输出
        return blk_in


if __name__ == '__main__':
    encoder = Encoder(d_model=4, num_layers=6, num_heads=2, num_ffn_hiddens=10, vocab_size=100)
    enc_in = torch.tensor([[1,2,3],
                          [3,4,5]], dtype=torch.int32)
    valid_lens = torch.tensor([2, 3], dtype=torch.int16)
    enc_out = encoder(enc_in, valid_lens)
    print(enc_out)