import torch
import pandas as pd
from data import get_zh_en_data_loader
from transformer import Transformer
import vocab


def train():
    # 超参数
    d_model = 512 # 模型向量维度
    num_epoch = 10 # 训练次数

    src_vocab = vocab.src_vocab() # 源语言词典
    tgt_vocab = vocab.tgt_vocab() # 目标语言词典



    model = Transformer(d_model=d_model, tgt_vocab_size=len(tgt_vocab))

    for epoch in range(num_epoch):
        data_loader = get_zh_en_data_loader()
        for batch_src, batch_tgt in data_loader:
        



    




if __name__ == '__main__':
    train()