"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
import math

import numpy as np
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : currently given sentences
    Key : sentences to check relationship with query
    Value : sentences that same with key
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout()

    def forward(self, q, k, v, mask=None, drop_prob=0.1):
        score = q @ k.transpose(-2, -1)  # dot product (Q * K^T)
        score /= math.sqrt(k.size()[0])  # divide by d_k^(1/2) for scaling

        if mask: score = score.masked_fill(mask, -np.inf)  # masking is option
        # prevents information from the sequence after the current word.

        score = self.softmax(score, dim=-1)  # pass them softmax
        score = self.dropout(score, drop_prob)  # apply dropout
        attentive_v = score @ v  # reflect score in Value
        return attentive_v
