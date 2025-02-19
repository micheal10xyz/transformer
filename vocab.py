import torch
import pandas as pd
import spacy

train_data_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'

df = pd.DataFrame([
    ['声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。', 'The statement added that Warren\'s colleagues were shocked and want him to turn himself in.'],
    ['不光改变硬件，软件也要跟上', 'We should not only change the hardware, but the software must also keep up.'],
    ['让各种文明和谐共存。', 'and allow all civilizations to coexist harmoniously.']
], columns=['0', '1'])
# df = pd.read_csv(train_data_path)

def src_vocab():
    vocab_set = set()
    tokenizer = spacy.load('zh_core_web_sm')
    for sentence in df['0']:
        tokens = tokenizer(sentence)
        token_texts = {token.text for token in tokens}
        vocab_set.update(token_texts)
    return {token: i for (i, token) in enumerate(vocab_set)}

def tgt_vocab():
    vocab_set = set()
    tokenizer = spacy.load('en_core_web_sm')
    for sentence in df['1']:
        tokens = tokenizer(sentence)
        token_texts = {token.text for token in tokens}
        vocab_set.update(token_texts)
    return {token: i for (i, token) in enumerate(vocab_set)}

    
# print(tgt_vocab())