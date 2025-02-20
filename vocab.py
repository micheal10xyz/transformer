import torch
import pandas as pd
from collections import Counter
import spacy
import thulac

train_data_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'

df = pd.DataFrame([
    ['声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。', 'The statement added that Warren\'s colleagues were shocked and want him to turn himself in.'],
    ['不光改变硬件，软件也要跟上', 'We should not only change the hardware, but the software must also keep up.'],
    ['让各种文明和谐共存。', 'and allow all civilizations to coexist harmoniously.']
], columns=['0', '1'])
# df = pd.read_csv(train_data_path)

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


def src_vocab():
    tokenizer = thulac.thulac(seg_only=True)
    tokens = []
    for sentence in df['0']:
        tokens = tokens + tokenizer.cut(sentence, True).split()
    return Vocab(tokens, reserved_tokens=['[pad]', '[bos]', '[eos]'])

def tgt_vocab():
    tokenizer = spacy.load('en_core_web_sm')
    tokens = []
    for sentence in df['1']:
       tokens = tokens + [token.text for token in tokenizer(sentence)]
    return Vocab(tokens, reserved_tokens=['[pad]', '[bos]', '[eos]'])

    
# test
# vocab = src_vocab()
# print(vocab.get_token_ids(['声明', '[unk]']))
