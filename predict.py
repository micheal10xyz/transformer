import torch
from torch import nn
from model import Transformer
from vocab import Vocab
import tokenizer

def predict(model: Transformer, src_sentence: str, src_vocab: Vocab, tgt_vocab: Vocab, num_steps: int, device: str=None):
    model.eval()
    # 构造编码器输入 
    src_tokens = tokenizer.tokenize_zh(src_sentence) + [src_vocab.eos]
    src_token_ids = src_vocab.get_token_ids(src_tokens)
    encoder_valid_lens = torch.tensor([len(src_token_ids)], device=device)
    src_token_ids += [src_vocab.pad_token_id] * (num_steps - len(src_token_ids))
    encoder_input = torch.tensor(src_token_ids, dtype=torch.long, device=device).unsqueeze(0) # shape [1, src_seq_len]
    
    # 构造解码器输入
    tgt_token_ids = [tgt_vocab.bos_token_id]
    decoder_input = torch.tensor(tgt_token_ids, dtype=torch.long, device=device).unsqueeze(0) # shape [1, tgt_seq_len]

    target_output = []
    # 循环调用大模型
    for _ in range(num_steps):
        predict_output = model(encoder_input=encoder_input, encoder_valid_lens=encoder_valid_lens, decoder_input=decoder_input, device=device)
        
    # 返回输出


if __name__ == '__main__':
    pass
    # predict(None, None, None, None)