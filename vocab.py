import torch
import pandas as pd
from collections import Counter
import tokenizer

train_data_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'
df = pd.read_csv(train_data_path)
df = df.head(10)

class Vocab:
    def __init__(self, tokens, min_freqs=0, reserved_tokens=None):
        """
        Args:
            tokens (list): token列表
            min_freqs (int): token出现的最小频率，当tokens中的token出现的频率小于该值，token会被丢弃
            reserved_tokens (list): 保留token列表，如 ['[pad]', '[bos]', '[eos]']
        """
        self.unk = '[unk]'
        self.id_to_token = [self.unk] + reserved_tokens
        self.token_to_id = {}
        counter = Counter(tokens)
        token_freqs = sorted(counter.items(), key = lambda item:item[1], reverse=True)
        for token, freqs in token_freqs:
            if freqs < min_freqs:
                continue
            self.id_to_token.append(token)
        self.token_to_id = {token: idx for (idx, token) in enumerate(self.id_to_token)}
    
       
    def get_token_ids(self, tokens):
        return [self.token_to_id.get(token, 0) for token in tokens]


    def get_tokens(self, ids):
        return map(lambda id: self.id_to_token[id] if id >= 0 and id < len(self.id_to_token) else self.unk)
    
    def __len__(self):
        return len(self.id_to_token)


def src_vocab():
    
    tokens = []
    for sentence in df['0']:
        tokens = tokens + tokenizer.tokenize_zh(sentence)
    return Vocab(tokens, reserved_tokens=['[pad]', '[bos]', '[eos]'])

def tgt_vocab():
    tokens = []
    for sentence in df['1']:
       tokens = tokens + tokenizer.tokenize_en(sentence)
    return Vocab(tokens, reserved_tokens=['[pad]', '[bos]', '[eos]'])

    
# test
if __name__ == '__main__':
    vocab = src_vocab()
    print(len(vocab))
