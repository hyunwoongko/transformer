"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device, requires_grad=False)

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        pos = pos / 10000 ** (_2i / d_model)
        # compute position information (same with original paper)

        self.encoding[:, 0::2] = torch.sin(pos)  # if 'i' is even [0, 2, 4, ... ] => sin
        self.encoding[:, 1::2] = torch.cos(pos)  # if 'i' is odd [1, 3, 5, ... ] => cos

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 31]

        return self.encoding[:seq_len, :]
        # [seq_len = 31, d_model = 512]
        # it will add tok_emb : [128, 31, 512]
