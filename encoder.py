import torch
from torch import nn
import positional_encoding

class Block(nn.Module):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, input, valid_lens):
        # 计算点积注意力
        # 残差连接，层归一化
        # 多层感知机
        # 残差连接，层归一化
        pass

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.encoder_block = [Block(d_model, num_heads) for i in range(num_layers)]


    
    def forward(self, input, valid_lens):
        """对input编码
        Args:
            input (Tensor): shape [batch_size, seq_len]，编码器输入
            valid_lens (Tensor): shape [batch_size]，有效长度
        """
        # 词嵌入（embedding）
        embeddings = self.embedding(input)
        # 添加位置编码
        block_input = self.positional_encoding(embeddings)
        # 循环执行编码器块
        for encoder_block in self.encoder_block:
            block_input = encoder_block(block_input, valid_lens)
        # 返回编码器输出
        return block_input


if __name__ == '__main__':
    encoder = Encoder(4, 100)
    input = torch.tensor([[1,2,3],
                          [3,4,5]], dtype=torch.int32)
    valid_lens = torch.Tensor([3,3])
    encoder(input[0], valid_lens)