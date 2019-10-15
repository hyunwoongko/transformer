"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from src.model.attention_layers.multi_head_attention import MultiHeadAttention
from src.model.other_layers.layer_normalization import LayerNormalization
from src.model.other_layers.position_wise_feed_forward import PositionWiseFeedForward
from src.model.other_layers.residual_Layer import ResidualLayer


class Encoder(nn.Module):

    def __init__(self, d_model, d_k, n_head):
        super(Encoder, self).__init__()
        """
        :param d_model: dimension of Model
        :param d_k: dimension of Key (such as Hidden Nodes)
        :param n_head: number of head for Multi-Head Attention
        """
        self.attention = MultiHeadAttention(d_model, d_k, n_head)
        self.normalization = LayerNormalization()
        self.feed_forward = PositionWiseFeedForward(d_model, d_k)
        self.residual = ResidualLayer()
        self.dropout = nn.Dropout()

    def forward(self, x):
        _x = x  # store input value for residual
        x = self.normalization(x)  # normalize
        x = self.attention(x, x, x)  # multi head attention
        x = self.residual(x, _x)  # add residual
        x = self.dropout(x)  # drop out

        _x = x  # store input value for residual
        x = self.normalization(x)  # normalize
        x = self.feed_forward(x)  # feed forward
        x = self.residual(x, _x)  # add residual
        x = self.dropout(x)  # drop out
        return x
