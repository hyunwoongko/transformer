"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn


class LayerNormalization(nn.Module):
    """
    Normalization for each layer
    """

    def __init__(self, e=1e-12):
        super(LayerNormalization, self).__init__()
        self.w_mean = nn.Parameter(torch.ones())
        self.w_std = nn.Parameter(torch.zeros())
        self.e = e  # to avoid situation that divided by zero

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.e)
