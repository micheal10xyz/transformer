import torch
from torch import nn
from model import Transformer
from vocab import Vocab
import tokenizer
import vocab
import pandas as pd
import time


def predict(model: Transformer, src_sentence: str, src_vocab: Vocab, tgt_vocab: Vocab, num_steps: int, device: str=None):
    model.eval()
    # 构造编码器输入 
    src_tokens = tokenizer.tokenize_zh(src_sentence) + [src_vocab.eos]
    src_token_ids = src_vocab.get_token_ids(src_tokens)
    enc_valid_lens = torch.tensor([len(src_token_ids)], device=device)
    src_token_ids += [src_vocab.pad_token_id] * (num_steps - len(src_token_ids))
    enc_in = torch.tensor(src_token_ids, dtype=torch.long, device=device).unsqueeze(0) # shape [1, src_seq_len]
    
    # 构造解码器输入
    tgt_token_ids = [tgt_vocab.bos_token_id]
    dec_in = torch.tensor(tgt_token_ids, dtype=torch.long, device=device).unsqueeze(0) # shape [1, tgt_seq_len]

    # 对源语言序列编码
    enc_out = model.encoder(enc_in, enc_valid_lens, device)

    # 初始化内部缓存，用来缓存历史时间步decoder-layer的输出，避免对token的重复解码，保证对一个token解码一次，再次使用，查询缓存
    dec_key_values_cache = model.decoder.init_dec_key_values_cache()

    target_output = []
    # 循环调用大模型
    for _ in range(num_steps):
        pred_out = model.decoder(dec_in, enc_out, enc_valid_lens, dec_key_values_cache, device)
        dec_in = pred_out.argmax(dim=2)
        pred_token_id = dec_in.squeeze(dim=0).type(torch.int)
        if pred_token_id == tgt_vocab.eos_token_id:
            break
        target_output.append(pred_token_id)
        print('pred token_id is ', pred_token_id)

    # 返回输出
    return ' '.join(tgt_vocab.get_tokens(target_output))


if __name__ == '__main__':
     # 超参数
    d_model = 64 # 模型向量维度
    num_epoch = 10 # 训练次数
    device = 'cpu' # Tensor or Module 计算用的设备
    num_layers = 6 # 编码器块，解码器块的个数
    num_heads = 4 # 多头注意力的头数
    num_ffn_hiddens = 256 # 前反馈层隐藏层的神经元个数
    lr = 1e-4 # 学习率

    src_vocab = vocab.src_vocab() # 源语言词典
    tgt_vocab = vocab.tgt_vocab() # 目标语言词典

    transformer = Transformer(d_model=d_model, num_layers=num_layers, num_heads=num_heads, num_ffn_hiddens=num_ffn_hiddens, src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
    transformer.to(device)
    checkpoint = torch.load('parameters/transformer_weights')
    transformer.load_state_dict(checkpoint)
    train_data_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'
    df = pd.read_csv(train_data_path)
    for src_sentence in df['0']:
        print(src_sentence)
        print(predict(transformer, src_sentence, src_vocab, tgt_vocab, 10))
        time.sleep(5)