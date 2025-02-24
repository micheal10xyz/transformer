import torch
import pandas as pd
from data import get_zh_en_data_loader
import model
import vocab
import tokenizer

def transform_sentence_to_token_ids(sentence, tokenize_func, vocab):
    """把句子变为token_ids，并在末尾填充句子结束标识

    Args:
        sentence (str): 句子，示例 '你好世界'
        tokenize_func (func): 分词器，用来对sentence分词
        vocab (Vocab): 词典
    """
    tokens = tokenize_func(sentence)
    token_ids = vocab.get_token_ids(tokens)
    # 添加句子结束标识
    token_ids += [vocab.eos_token_id]
    return token_ids


def transform_sentences_to_mini_batch(sententces, tokenize_func, vocab):
    """把句子变成小批量数据集

    Args:
        sententces (list): 句子列表
        tokenize_func (func): 分词函数，用来对句子分词
        vocab (Vocab): 词典

    Returns:
        (batch_token_ids, valid_lens): batch_token_ids tensor shape [batch_size, seq_len] 小批量token_ids； valid_lens tensor shape [batch_size] 以及有效的token长度列表，后续用来计算掩码
    """
    batch_token_ids = [transform_sentence_to_token_ids(sentence, tokenize_func, vocab) for sentence in sententces]
    # 计算句子有效长度
    valid_lens = [len(token_ids) for token_ids in batch_token_ids]
    max_valid_len = max(valid_lens)
    # 填充'<pad>'
    batch_token_ids = [token_ids + [vocab.pad_token_id] * (max_valid_len - len(token_ids)) for token_ids in batch_token_ids]
    return torch.Tensor(batch_token_ids), torch.Tensor(valid_lens)


def train():
    # 超参数
    d_model = 512 # 模型向量维度
    num_epoch = 1 # 训练次数
    device = 'cpu' # Tensor or Module 计算用的设备

    src_vocab = vocab.src_vocab() # 源语言词典
    tgt_vocab = vocab.tgt_vocab() # 目标语言词典

    transformer = model.Transformer(d_model=d_model, tgt_vocab_size=len(tgt_vocab))
    transformer.to(device)
    transformer.train()
    for epoch in range(num_epoch):
        data_loader = get_zh_en_data_loader()
        for batch_src, batch_tgt in data_loader:
            src_batch_token_ids, src_valid_lens = transform_sentences_to_mini_batch(batch_src, tokenizer.tokenize_zh, src_vocab)
            tgt_batch_token_ids, tgt_valid_lens = transform_sentences_to_mini_batch(batch_tgt, tokenizer.tokenize_en, tgt_vocab)

            # 解码器输入
            decoder_input = torch.Tensor([tgt_vocab.bos_token_id] * tgt_batch_token_ids.shape[0]).reshape(-1, 1)
            # Teacher Forcing, 将真实的目标序列作为解码器输入，代替模型的预测输出，加速模型收敛速度
            decoder_input = torch.cat((decoder_input, tgt_batch_token_ids[:, :-1]), dim=1)
            print(decoder_input)

            predict_output = transformer(src_batch_token_ids, src_valid_lens, decoder_input, tgt_valid_lens)

            # todo
    




if __name__ == '__main__':
    train()