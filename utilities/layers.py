import torch.nn as nn


def init_conv_weights(layer):
    nn.init.normal(layer.weight.data, std=0.01)
    nn.init.constant(layer.bias.data, val=0)
    return layer


def conv1x1(in_channels, out_channels, **kwargs):

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3(in_channels, out_channels, **kwargs):

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer


