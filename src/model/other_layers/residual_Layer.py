"""
@author : Hyunwoong
@when : 8/27/2019
@homepage : https://github.com/gusdnd852
"""

from torch import nn


class ResidualLayer(nn.Module):
    """
    add two tensor, this layer is layer for increasing code readability
    reference : https://arxiv.org/abs/1512.0a3385
    (Deep Residual Learning for Image Recognition, Kaiming He and others - 2015)
    """

    def forward(self, x, _x):
        return x + _x
