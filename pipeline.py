import torch
import torch.nn as nn

def Pipeline(blocks):
    return nn.Sequential(*blocks)

def Id():
    return nn.Identity()
