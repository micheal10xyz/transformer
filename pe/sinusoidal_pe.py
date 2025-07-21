import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=1000):
        """初始化函数
        Args:
            dim (int): 位置张量维度
            max_len (int, optional): 最大位置，默认1000.
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float16).unsqueeze(1)
        base = 10000
        base_pow = torch.pow(base, torch.arange(0, max_len, 2, dtype=torch.float16) / dim).unsqueeze(0)
        self.pe[::, 0::2] = torch.sin(pos / base_pow)
        self.pe[::, 1::2] = torch.cos(pos / base_pow)
    
    def forward(self, X: torch.Tensor):
        """
        Args:
            X (tensor): 小批量词嵌入张量，shape为(b, n, dim)， b为小批量大小，n为token的数量，dim为token embedding的维度
        """
        return X + torch.unsqueeze(self.pe[0:X.shape[1], ::])
    

def num_to_binary(num, bits):
    bins = []
    bit = 1
    for i in range(bits):
        bins.append(1 if num &bit > 0 else 0)
        bit = bit << 1
    return bits


def num_bits_tensor(num, bits):
    return torch.tensor([num_to_binary(i, bits) for i in range(num)])


plt.figure(figsize=(10,6))
plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
plt.xlabel('Encoding Dimension')
plt.ylabel('Token Position')
plt.colorbar()
plt.title('Positional Encoding Heatmap')
plt.show()