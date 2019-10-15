"""
@author : Hyunwoong
@when : 8/27/2019
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.sin = None
        self.cos = None

    def forward(self, x):
        return x
