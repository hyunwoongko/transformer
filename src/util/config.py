"""
@author : Hyunwoong
@when : 8/21/2019
@homepage : https://github.com/gusdnd852
"""
import torch

# Pytorch Device (CUDA or CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
