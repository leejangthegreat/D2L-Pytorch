import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from typing import Sequence
from lenet import train
def VGG_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a VGG block with multiple conv layers"""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def VGG(conv_arch: Sequence[tuple[int, int]]) -> nn.Sequential:
    """Build VGG-11"""
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

def VGG96(conv_arch: Sequence[tuple[int, int]]) -> nn.Sequential:
    """VGG for 96 x 96 on FashionMNIST"""
    conv_blks: list = []
    in_channels: int = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(in_channels * 3 * 3, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 10)
    )


if __name__ == '__main__':
    conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))  # A small arch of VGG-11
    net = VGG(conv_arch)
    net96= VGG96(conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 128
    trainIter, testIter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    trainIter96, testIter96 = d2l.load_data_fashion_mnist(batch_size, resize=96)
    # train(net, trainIter, testIter, num_epochs, lr)
    train(net96, trainIter96, testIter96, num_epochs, lr)
