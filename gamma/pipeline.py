import torch
import torch.nn as nn

def Pipeline(blocks):
    return nn.Sequential(*blocks)

def LambdaPow(lbd, n):
    return nn.Sequential(*[lbd() for _ in range(n)])
