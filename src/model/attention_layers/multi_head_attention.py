"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from src.model.attention_layers.scale_dot_product_attention import ScaleDotProductAttention
from src.util.config import device


class MultiHeadAttention(nn.Module):
    """
    compute multi head attention (multiple attention)
    """

    def __init__(self, d_model, d_k, n_head):
        super(MultiHeadAttention, self).__init__()
        """
        :param d_model: dimension of Model
        :param d_k: dimension of Key (such as Hidden Nodes)
        :param n_head: number of head for Multi-Head Attention
        """
        self.d_model = d_model
        self.d_k = d_k
        self.n_head = n_head
        self.attention = ScaleDotProductAttention().to(device)

        """
        linear layer (matrix multiplication with weight) 
        for query, key, value, concatenation
        """
        self.w_q = nn.Linear(d_model, d_k).to(device)  # weight vector for Query
        self.w_k = nn.Linear(d_model, d_k).to(device)  # weight vector for Key
        self.w_v = nn.Linear(d_model, d_k).to(device)  # weight vector for Value
        self.w_concat = nn.Linear(n_head * d_k, d_model).to(device)
        # weight vector for concatenation all attention vector

    def forward(self, q, k, v, mask=None, drop_prob=0.1):
        # 1) check current batch size
        batch_size = q.size()[0]

        # 2) matrix multiplication and linear projection
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = [self.multi_head(each, batch_size) for each in [q, k, v]]

        # 3) apply attention (masking is option)
        if mask: mask = mask.repeat(self.n_head, 1, 1)
        attentive_input = self.attention(q, k, v, mask)

        # 4) concat all attentive inputs and multiply with w_concat
        attentive_input = attentive_input.view(batch_size, -1).to(device)
        attentive_input = self.w_concat(attentive_input)
        return attentive_input

    def multi_head(self, tensor, batch_size):
        # 1) create multi-head tensor
        multi_head = torch.zeros(self.n_head, batch_size, self.d_k, device=device)

        # 2) assign tensor to multi-head tensor
        for n in range(self.n_head): multi_head[n] = tensor
        return multi_head
