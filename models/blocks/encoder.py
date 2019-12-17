"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm = LayerNorm(d_model=d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        _x = x
        x = self.attention(x, x, x)
        x += x
        x = self.norm(x)

        _x = x
        x = self.feed_forward(x)
        x += x
        x = self.norm(x)

        out = self.drop_out(x)
        return out
