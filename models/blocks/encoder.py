"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.multi_head_attention = None
        self.layer_normalization = nn.LayerNorm()
        self.feed_forward = None
        self.drop_out = nn.Dropout(p=None)


    def forward(self, x):
        shortcut = x
        x = self.multi_head_attention(x, x, x)
        x += shortcut
        x = self.layer_normalization(x)

        shortcut = x
        x = self.feed_forward(x)
        x += shortcut
        x = self.layer_normalization(x)
        return x
