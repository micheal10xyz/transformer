import torch
from torch import nn
import positional_encoding
import attention
import ffn
import math


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_ffn_hiddens, blk_no, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blk_no = blk_no
        self.self_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.position_wise_ffn = ffn.PositionWiseFFN(d_model, num_ffn_hiddens, d_model)

    
    def forward(self, dec_in, enc_out, enc_valid_lens, dec_key_values_cache):
        # 计算dec_in掩码, 训练过程是在一个时间步（time step）完成的，需要防止在计算自注意力时关注到未来的信息，需要对注意力的上三角添加掩码
        # 在推理阶段，因为自回归的，每个时间步预测一个token，不存在未来信息，不需要添加掩码
        if self.training:
            dec_in_valid_lens = torch.arange(1, dec_in.shape[1] + 1).repeat(dec_in.shape[0], 1)
        else:
            dec_in_valid_lens = None
        
        if dec_key_values_cache is None:
            key_values = dec_in
        else:
            key_values = dec_key_values_cache[self.blk_no]
            key_values = dec_in if key_values is None else torch.cat((key_values, dec_in), dim=1)
            dec_key_values_cache[self.blk_no] = key_values

        # 对dec_in计算自注意力
        dec_self_attention = self.self_attention(dec_in, key_values, key_values, dec_in_valid_lens)
        # 残差连接，层归一化
        queries = self.norm1(dec_in + dec_self_attention)
        # 计算encoder-decoder注意力
        enc_dec_attention = self.encoder_decoder_attention(queries, enc_out, enc_out, enc_valid_lens)
        # 残差连接，层归一化
        ffn_in = self.norm2(queries + enc_dec_attention)
        # 前馈神经网络
        ffn_out = self.position_wise_ffn(ffn_in)
        # 残差连接，层归一化
        return self.norm3(ffn_in + ffn_out)


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block_' + str(i), Block(d_model, num_heads, num_ffn_hiddens, i))
        self.num_layers = num_layers
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, dec_in, enc_out, enc_valid_lens, dec_key_values_cache=None):
        """解码器解码

        Args:
            dec_in (Tensor): shape [batch_size, tgt_seq_len] 解码器输入
            enc_out (Tensor): shape [batch_size, src_seq_len, d_model] 编码器的输出
            enc_valid_lens (Tensor) shape [batch_size]
        """
        # 对dec_in embedding
        embedings = self.embedding(dec_in) * math.sqrt(self.d_model) # shape [batch_size, tgt_seq_len, d_model]
        # 添加位置编码
        blk_in = self.positional_encoding(embedings)
        # decoder block 
        for i, blk in enumerate(self.blks):
            blk_in = blk(blk_in, enc_out, enc_valid_lens, dec_key_values_cache)
        # 计算logits
        return self.linear(blk_in)
    

    def init_dec_key_values_cache(self):
        """初始化解码器层输入缓存，在预测阶段使用

        Returns:
            list[Tensor]: 列表大小固定为num_layers, 第i个decoder layer对应的缓存是list的第i个元素
        """
        return [None] * self.num_layers
