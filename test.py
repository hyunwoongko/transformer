"""
@author : Hyunwoong
@when : 2019-12-19
@homepage : https://github.com/gusdnd852
"""

import math
from collections import Counter

import numpy as np

from data import *
from models.model.transformer import Transformer
from util.bleu import get_bleu, idx_to_word


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=0.00,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')


def test_model(num_examples):
    iterator = test_iter
    model.load_state_dict(torch.load("./saved/model-saved.pt"))

    with torch.no_grad():
        batch_bleu = []
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])

            total_bleu = []
            for j in range(num_examples):
                try:
                    src_words = idx_to_word(src[j], loader.source.vocab)
                    trg_words = idx_to_word(trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)

                    print('source :', src_words)
                    print('target :', trg_words)
                    print('predicted :', output_words)
                    print()
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            print('BLEU SCORE = {}'.format(total_bleu))
            batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))


if __name__ == '__main__':
    test_model(num_examples=batch_size)
