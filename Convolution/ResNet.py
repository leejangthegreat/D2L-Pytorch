"""
Residual Block and ResNet.
"""

import torch
from torch import nn
from torch.nn.functional import relu
class Residual(nn.Module):
    def __init__(self, input_channels, out_channels, use_res_conv1=False, strides=1):
        super().__init__()
        self.main_stream = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.res_conv = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides) if use_res_conv1 else None

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        Y = self.main_stream(X)
        if not self.res_conv is None:
            X = self.res_conv(X)
        Y += X
        return relu(Y)
