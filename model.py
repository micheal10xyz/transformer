import torch
from torch import nn
import encoder
import decoder

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_ffn_hiddens, src_vocab_size, tgt_vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder.Encoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads, num_ffn_hiddens=num_ffn_hiddens, vocab_size=src_vocab_size)
        self.decoder = decoder.Decoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads, num_ffn_hiddens=num_ffn_hiddens, vocab_size=tgt_vocab_size)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    
    def forward(self, encoder_input, encoder_valid_lens, decoder_input, device=None):
       encoder_output = self.encoder(encoder_input, encoder_valid_lens, device)
       decoder_output = self.decoder(encoder_output, encoder_valid_lens, decoder_input, device)
       # 计算logits
       return self.linear(decoder_output)