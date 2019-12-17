"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder import Encoder
from models.embedding.transformer_embedding import TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model,
                 ffn_hidden, n_layers, n_head, max_len, drop_prob, device):
        super(Transformer, self).__init__()
        self.enc_embedding = TransformerEmbedding(vocab_size=enc_voc_size,
                                                  d_model=d_model,
                                                  max_len=max_len,
                                                  drop_prob=drop_prob,
                                                  device=device)

        self.encoders = nn.Sequential(*[Encoder(d_model=d_model,
                                                drop_prob=drop_prob,
                                                ffn_hidden=ffn_hidden,
                                                n_head=n_head) for _ in range(n_layers)])

    def forward(self, source, target):
        source = self.enc_embedding(source)
        source = self.encoders(source)
        print(source.size())

        return source
