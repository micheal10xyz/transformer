import torch
from torch import nn

class PositionWiseFFN(nn.Module):
    def __init__(self,num_inputs, num_hiddens, num_outputs, *args, **kwargs):
        """基于位置的前馈神经网络

        Args:
            num_inputs (int): 输入特征维度
            num_hiddens (int): 隐藏层特征维度
            num_outputs (int): 输出特征维度
        """
        super().__init__(*args, **kwargs)
        self.dense0 = nn.Linear(num_inputs, num_hiddens)    # 线型层
        self.relu = nn.ReLU()   # 激活函数
        self.dense1 = nn.Linear(num_hiddens, num_outputs)   # 线型层


    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): shape [*, num_inputs] 1D 或2D 或3D 或高维张量，最后一维的大小必须是num_inputs, tensor.shape(-1) == num_inputs

        Returns:
            Tensor: shape [*, num_inputs] 1D 或2D 或3D 或高维张量，最后一维的大小是num_outputs
        """
        return self.dense1(self.relu(self.dense0(inputs)))
