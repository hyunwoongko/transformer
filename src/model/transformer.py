"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from src.model.main_layers.encoder import Encoder


class Transformer(nn.Module):
    """
    The Transformer Model (Encoder Decoder Network using only Attention)
    reference : https://arxiv.org/abs/1706.03762
    (Attention is All You Need, Google Brain - 2017)
    """

    def __init__(self, d_model, d_k, n_head=8):
        """
        :param d_model: dimension of Model
        :param d_k: dimension of Key (such as Hidden Nodes)
        :param n_head: number of head for Multi-Head Attention
        """
        super(Transformer, self).__init__()
        self.encoder1 = Encoder(d_model, d_k, n_head)

    def forward(self, x):
        x = self.encoder1(x)
        return x
