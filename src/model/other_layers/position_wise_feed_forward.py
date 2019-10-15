"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionWiseFeedForward(nn.Module):
    """
    Layer for Feed Forward
    """

    def __init__(self, d_model, d_k):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_k)
        self.fc2 = nn.Linear(d_k, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, drop_prob=0.1):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x, p=drop_prob)
        x = self.fc2(x)
        return x
