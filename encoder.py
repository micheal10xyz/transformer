import torch
from torch import nn
import positional_encoding
import attention
import ffn

class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_ffn_hiddens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention.MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.position_wise_ffn = ffn.PositionWiseFFN(d_model, num_ffn_hiddens, d_model)


    def forward(self, input, valid_lens, device:None):
        """编码器块

        Args:
            input (Tensor): shape [batch_size, seq_len, d_model]
            valid_lens (_type_): shape [batch_size]
        """
        # 计算点积注意力
        attention = self.attention(input, input, input, valid_lens, device)
        # 残差连接，层归一化
        ffn_input = self.norm(input + attention)
        # 多层感知机(mlp)
        ffn_output = self.position_wise_ffn(ffn_input)
        # 残差连接，层归一化
        return self.norm(ffn_input + ffn_output)


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.encoder_block = [Block(d_model, num_heads, num_ffn_hiddens) for _ in range(num_layers)]


    
    def forward(self, input, valid_lens, device: None):
        """对input编码
        Args:
            input (Tensor): shape [batch_size, seq_len]，编码器输入
            valid_lens (Tensor): shape [batch_size]，有效长度
            device(str): 在device运行存储张量
        """
        # 词嵌入（embedding）
        embeddings = self.embedding(input)
        # 添加位置编码
        block_input = self.positional_encoding(embeddings)
        # 循环执行编码器块
        for block in self.encoder_block:
            block_input = block(block_input, valid_lens, device)
        # 返回编码器输出
        return block_input


if __name__ == '__main__':
    encoder = Encoder(d_model=4, num_layers=6, num_heads=2, num_ffn_hiddens=10, vocab_size=100)
    input = torch.tensor([[1,2,3],
                          [3,4,5]], dtype=torch.int32)
    valid_lens = torch.tensor([2, 3], dtype=torch.int16)
    encoder_output = encoder(input, valid_lens, None)
    print(encoder_output)