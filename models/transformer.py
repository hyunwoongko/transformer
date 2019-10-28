"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.transformer_embedding import TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model, max_len, drop_prob, device):
        super(Transformer, self).__init__()
        self.enc_embedding = TransformerEmbedding(vocab_size=enc_voc_size,
                                                  d_model=d_model,
                                                  max_len=max_len,
                                                  drop_prob=drop_prob,
                                                  device=device)

        self.dec_embedding = TransformerEmbedding(vocab_size=dec_voc_size,
                                                  d_model=d_model,
                                                  max_len=max_len,
                                                  drop_prob=drop_prob,
                                                  device=device)

    def forward(self, source, target):
        source = self.enc_embedding(source)
        return source
