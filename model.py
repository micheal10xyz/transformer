import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.decoder = None
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src (torch.Tensor): 源序列，shape [batch_size, seq_len]
            tgt (torch.Tensor): 目标序列，shape [batch_size, seq_len]
            src_mask (torch.Tensor): 源序列掩码，shape [batch_size, seq_len, seq_len]
            tgt_mask (torch.Tensor): 目标序列掩码，shape [batch_size, seq_len, seq_len]
        """
        enc_output = self.encoder(src, src_mask)
        pass