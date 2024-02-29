"""
Realization of Inception Block and other structures in GoogleNet.
Running this module directly will train GoogleNet on fashion-MNIST.
The training is with cpu only for now.
"""

import torch
from torch import nn
from d2l import torch as d2l
from typing import Sequence
class Inception(nn.Module):
    """Inception Block for googleNet. Four parallel paths with different conv kernel."""
    def __init__(self,
                 in_channels: int, channel_1: int, channel_2: Sequence[int],
                 channel_3: Sequence[int], channel_4: int,
                 **kwargs):
        """Original structure for Inception."""
        super().__init__(**kwargs)
        # Path 1: sngle 1x1 conv layer
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, channel_1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, channel_2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(channel_2[0], channel_2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, channel_3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(channel_3[0], channel_3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, channel_4, kernel_size=1), nn.ReLU()
        )

    def forward(self, X):
        return torch.cat((self.p1(X), self.p2(X), self.p3(X), self.p4(X)), dim=1)
