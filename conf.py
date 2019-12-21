"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 32
max_len = 50
d_model = 1024
n_layers = 8
n_heads = 8
ffn_hidden = 4096
drop_prob = 0.3

# optimizer parameter setting
init_lr = 1e-6
factor = 0.8
min_lr = init_lr * 1e-12
patience = 7
warmup = 300
weight_decay = 5e-5
epoch = 2000
clip = 1
inf = float('inf')
