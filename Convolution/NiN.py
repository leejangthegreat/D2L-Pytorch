from torch import nn
from d2l import torch as d2l
from lenet import train
def NiN_block(in_channels: int, out_channels: int, kernel_size: int, strides: int, padding: int) -> nn.Sequential:
    """Build a NiN block with one conv layer and two 1x1 conv layer"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

NiN_net = nn.Sequential(
    NiN_block(1, 96, 11, 4, 0),  # 54x54
    nn.MaxPool2d(kernel_size=3, stride=2),  # 26x26
    NiN_block(96, 256, 5, 1, 2),  # 26x26
    nn.MaxPool2d(kernel_size=3, stride=2),  # 12x12
    NiN_block(256, 384, 3, 1, 1),  # 12x12
    nn.MaxPool2d(kernel_size=3, stride=2),  # 5x5
    nn.Dropout(0.5),
    NiN_block(384, 10, 3, 1, 1),  # Use NiN block instead of Linear layer. 5x5
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

NiN_net96 = nn.Sequential(
    NiN_block(1, 96, 7, 2, 0),  # 45x45
    nn.MaxPool2d(kernel_size=3, stride=2),  # 22x22
    NiN_block(96, 256, 5, 1, 2),  # 22x22
    nn.MaxPool2d(kernel_size=3, stride=2),  # 10x10
    NiN_block(256, 384, 3, 1, 1),  # 10x10
    nn.MaxPool2d(kernel_size=3, stride=2), # 4x4
    nn.Dropout(0.5),
    NiN_block(384, 10, 3, 1, 1),  # 4x4
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    train(NiN_net96, train_iter, test_iter, num_epochs, lr)

