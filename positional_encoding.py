import torch
from torch import nn

class PositionEncoding(nn.Module):
    def __init__(self,d_model, max_pos, *args, **kwargs):
        """正余弦位置编码

        Args:
            d_model (int): 向量维度
            max_position (int): 可以支持的最大位置
        """
        super().__init__(*args, **kwargs)
        pos = torch.arange(max_pos).reshape(-1, 1)
        # print('pos ', pos)
        exponent = torch.arange(d_model / 2) * 2 / d_model
        # print('exponent ', exponent)
        base = 10000
        params = pos / torch.pow(base, exponent)
        # print('params is ', params)

        self.pe = torch.zeros(max_pos, d_model)

        self.pe[:, 0::2] = torch.sin(params)
        self.pe[:, 1::2] = torch.cos(params)
        # print('pe is ', self.pe)


    def forward(self, X):
        """

        Args:
            X (Tensor): shape [batch_size, seq_len, d_model]
        """
        return X + self.pe[:X.shape[1]]


if __name__ == '__main__':
    position_encoding = PositionEncoding(4, 10)
    X = torch.ones(4, 4)
    X = torch.unsqueeze(X, 0)
    print(position_encoding(X))
    