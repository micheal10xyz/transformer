import torch
import pandas as pd
from data import get_zh_en_data_loader
import model
import vocab
import tokenizer
from torch import nn

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
    return torch.tensor(batch_token_ids, dtype=torch.int), torch.tensor(valid_lens, dtype=torch.int)


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(reduction='none')
    

    def forward(self, input: torch.Tensor, target: torch.Tensor, valid_lens: torch.Tensor, device=None):
        """
        掩码交叉熵损失
        Args:
            input (torch.Tensor): shape [batch_size, seq_len, vocab_size] 预测值
            target (torch.Tensor): shape [batch_size, seq_len] 目标值
            valid_lens (torch.Tensor): shape [batch_size] 目标值有效长度
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        mask = torch.arange(seq_len, device=device).repeat(batch_size, 1) >= valid_lens.reshape(-1, 1)
        l = self.loss(input.transpose(1, 2), target)
        return l[~mask].mean()



def train():
    # 超参数
    d_model = 512 # 模型向量维度
    num_epoch = 3 # 训练次数
    device = 'cpu' # Tensor or Module 计算用的设备
    num_layers = 6 # 编码器块，解码器块的个数
    num_heads = 8 # 多头注意力的头数
    num_ffn_hiddens = 128 # 前反馈层隐藏层的神经元个数
    lr = 1e-4 # 学习率

    src_vocab = vocab.src_vocab() # 源语言词典
    tgt_vocab = vocab.tgt_vocab() # 目标语言词典

    transformer = model.Transformer(d_model=d_model, num_layers=num_layers, num_heads=num_heads, num_ffn_hiddens=num_ffn_hiddens, src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
    # todo 权重初始化，加速收敛
    transformer.to(device)
    transformer.train()
    # 优化算法
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    # 损失函数（目标函数）
    loss = MaskedCrossEntropyLoss()
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        data_loader = get_zh_en_data_loader()
        epoch_loss = 0
        for batch_src, batch_tgt in data_loader:
            src_batch_token_ids, src_valid_lens = transform_sentences_to_mini_batch(batch_src, tokenizer.tokenize_zh, src_vocab)
            tgt_batch_token_ids, tgt_valid_lens = transform_sentences_to_mini_batch(batch_tgt, tokenizer.tokenize_en, tgt_vocab)

            # 解码器输入
            decoder_input = torch.tensor([tgt_vocab.bos_token_id] * tgt_batch_token_ids.shape[0], dtype=torch.long, device=device).reshape(-1, 1)
            # Teacher Forcing, 将真实的目标序列作为解码器输入，代替模型的预测输出，加速模型收敛速度
            decoder_input = torch.cat((decoder_input, tgt_batch_token_ids[:, :-1]), dim=1)
            
            # 调用模型，预测
            predict_output = transformer(src_batch_token_ids, src_valid_lens, decoder_input, device)

            # 计算损失
            l = loss(predict_output, decoder_input, tgt_valid_lens, device)

            # 计算梯度
            l.backward()

            # 更新梯度
            optimizer.step()
            # 计算训练周期中的loss
            epoch_loss += l
            print(f'epoch {epoch} loss {l}')
        print(f'epoch No {epoch}, loss is {epoch_loss}')

    




if __name__ == '__main__':
    train()