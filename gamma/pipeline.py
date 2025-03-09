import torch
import torch.nn as nn

def Pipeline(blocks):
    return nn.Sequential(*blocks)


def RepeatBlock(block, n):
    return nn.Sequential(*[block() for _ in range(n)])
