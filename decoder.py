import torch
from torch import nn
import positional_encoding
import attention
import ffn


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_ffn_hiddens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.position_wise_ffn = ffn.PositionWiseFFN(d_model, num_ffn_hiddens, d_model)

    
    def forward(self, enc_in, enc_valid_lens, dec_in, dec_in_valid_lens, device=None):
        # 对dec_in计算自注意力
        self_attention = self.self_attention(dec_in, dec_in, dec_in, dec_in_valid_lens, device)
        queries = self.norm(dec_in + self_attention)
        # 计算encoder-decoder注意力
        encoder_decoder_attention = self.encoder_decoder_attention(queries, enc_in, enc_in, enc_valid_lens, device)
        ffn_input = self.norm(queries + encoder_decoder_attention)
        # ffn
        ffn_output = self.position_wise_ffn(ffn_input)
        return self.norm(ffn_input + ffn_output)


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding.PositionEncoding(d_model, max_pos=1000)
        self.dec_blks = [Block(d_model, num_heads, num_ffn_hiddens) for _ in range(num_layers)]
    
    def forward(self, enc_in, enc_valid_lens, dec_in, device:None):
        """解码器解码

        Args:
            enc_in (Tensor): shape [batch_size, src_seq_len, d_model] 编码器的输出
            dec_in (Tensor): shape [batch_size, tgt_seq_len] 解码器输入
        """
        # 对dec_in embedding
        embedings = self.embedding(dec_in) # shape [batch_size, tgt_seq_len, d_model]
        # 添加位置编码
        blk_in = self.positional_encoding(embedings)
        # 计算dec_in掩码
        dec_in_valid_lens = torch.arange(1, dec_in.shape[1] + 1, device=device).repeat(dec_in.shape[0], 1)
        # decoder block 
        for blk in self.dec_blks:
            blk_in = blk(enc_in, enc_valid_lens, blk_in, dec_in_valid_lens)

        return blk_in
