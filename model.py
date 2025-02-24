import torch
from torch import nn
import encoder

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, src_vocab_size, tgt_vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder.Encoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads, num_ffn_hiddens=num_ffn_hiddens, vocab_size=src_vocab_size)
        self.decoder = None
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    
    def forward(self, encoder_input, encoder_valid_lens, decoder_input):
       # todo 编码器输入掩码
       # todo 编码器输出
       # todo 解码器输入掩码
       # todo 解码器输出
       # todo 计算logits
       pass