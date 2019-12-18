"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 50
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 256
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-4
factor = 0.8
patience = 10
warmup = 5
weight_decay = 5e-4
epoch = 2000
clip = 1
inf = float('inf')
