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

    
    def forward(self, encoder_output, decoder_input, decoder_input_valid_lens):
        # 对decoder_input计算自注意力
        self_attention = self.attention(decoder_input, decoder_input, decoder_input)
        # 计算encoder-decoder注意力
        # ffn


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.decoder_blocks = [Block(d_model, num_heads, num_ffn_hiddens) for _ in range(num_layers)]
    
    def forward(self, encoder_output, decoder_input):
        """解码器解码

        Args:
            encoder_output (Tensor): shape [batch_size, src_seq_len, d_model] 编码器的输出
            decoder_input (Tensor): shape [batch_size, tgt_seq_len] 解码器输入
        """
        # 对decoder_input embedding
        embedings = self.embedding(decoder_input) # shape [batch_size, tgt_seq_len, d_model]
        # 添加位置编码
        block_input = self.positional_encoding(embedings)
        # decoder block 
        for block in self.decoder_blocks:
            block_input = block(encoder_output, block_input)

        return block_input
