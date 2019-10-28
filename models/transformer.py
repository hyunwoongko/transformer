"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.transformer_embedding import TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model, drop_prob):
        super(Transformer, self).__init__()
        self.enc_embedding = TransformerEmbedding(vocab_size=enc_voc_size,
                                                  d_model=d_model,
                                                  drop_prob=drop_prob)

        self.dec_embedding = TransformerEmbedding(vocab_size=dec_voc_size,
                                                  d_model=d_model,
                                                  drop_prob=drop_prob)

    def forward(self, x):
        x = self.enc_embedding(x)
        return x
