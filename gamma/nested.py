import torch.nn as nn
from gamma.activations import _act

def build_network(shapes, layer_constructor, activation='relu', **layer_kwargs):
    layers = []
    for i in range(len(shapes) - 1):
        layers.append(layer_constructor(shapes[i], shapes[i + 1], **layer_kwargs))
        if i < len(shapes) - 2:
            layers.append(_act(activation))
    return nn.Sequential(*layers)

def MLP(shapes, activation='relu'):
    return build_network(shapes, nn.Linear, activation=activation)

def ConvNet(channels, activation='relu', kernel_size=3, stride=1, padding=1):
    return build_network(
        channels, 
        nn.Conv2d, 
        activation=activation,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
