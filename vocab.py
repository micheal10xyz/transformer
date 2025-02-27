import torch
import pandas as pd
from collections import Counter
import tokenizer

train_data_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'
df = pd.read_csv(train_data_path)

class Vocab:
    def __init__(self, tokens, min_freqs=0):
        """
        Args:
            tokens (list): token列表
            min_freqs (int): token出现的最小频率，当tokens中的token出现的频率小于该值，token会被丢弃
        """
        self.unk = '<unk>'  # 未知token
        self.pad = '<pad>'  # 填充token
        self.bos = '<bos>'  # 句子开始token
        self.eos = '<eos>'  # 句子结束token
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.id_to_token = [self.unk, self.pad, self.bos, self.eos]
        self.token_to_id = {}
        counter = Counter(tokens)
        token_freqs = sorted(counter.items(), key = lambda item:item[1], reverse=True)
        for token, freqs in token_freqs:
            if freqs < min_freqs:
                continue
            self.id_to_token.append(token)
        self.token_to_id = {token: idx for (idx, token) in enumerate(self.id_to_token)}
    
       
    def get_token_ids(self, tokens):
        """根据token获取token_id

        Args:
            tokens (list): token列表，示例 ['你好', '世界']

        Returns:
            list: token_id列表，示例 [1, 2]
        """
        return [self.token_to_id.get(token, 0) for token in tokens]


    def get_tokens(self, ids):
        return [self.id_to_token[id] if id >= 0 and id < len(self.id_to_token) else self.unk for id in ids]
        # return map(lambda id: self.id_to_token[id] if id >= 0 and id < len(self.id_to_token) else self.unk, ids)
    
    def __len__(self):
        return len(self.id_to_token)


def src_vocab():
    
    tokens = []
    for sentence in df['0']:
        tokens = tokens + tokenizer.tokenize_zh(sentence)
    return Vocab(tokens)

def tgt_vocab():
    tokens = []
    for sentence in df['1']:
       tokens = tokens + tokenizer.tokenize_en(sentence)
    return Vocab(tokens)

    
# test
if __name__ == '__main__':
    token_ids = [57, 299]
    vocab = tgt_vocab()
    tokens = vocab.get_tokens(token_ids)
    print(len(vocab))
    print(tokens)
    print(' '.join(tokens))
    
