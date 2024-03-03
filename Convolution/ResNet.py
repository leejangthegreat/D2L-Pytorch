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

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    """
    Generate a residual block.
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_res_conv1=True, strides=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return blk

resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    *resnet_block(64, 64, 2, first_block=True),
    *resnet_block(64, 128, 2),
    *resnet_block(128, 256, 2),
    *resnet_block(256, 512, 2),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(512, 10)
)

if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in resnet:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
