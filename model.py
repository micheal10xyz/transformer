import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.decoder = None
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    
    def forward(self, encoder_input, encoder_valid_lens, decoder_input):
       # todo 编码器输入掩码
       # todo 编码器输出
       # todo 解码器输入掩码
       # todo 解码器输出
       # todo 计算logits
       pass