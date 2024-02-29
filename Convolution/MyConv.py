"""
Self-define convolutional function and layer, on 2D metrix
Also with an example of learning convolution kernel.

Usually, Kernel of one conv layer include multiple k with channel dimension.
Each k evaluates correlation with an input X of multiple channels, and result in an output of one channel.
all output channels from each k in 4D Kernel stack and get multiple channels output.
"""

import torch
from torch import nn
from d2l import torch as d2l
from typing import Sequence
def corr2d(X, kernel):
    """2D correlation"""
    r, c = kernel.shape
    Y = torch.zeros((X.shape[0] - r + 1, X.shape[1] - c + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + r, j:j + c] * kernel).sum()
    return Y

class Conv2D(nn.Module):
    """Convolution Layer"""
    def __init__(self, kernel_size: Sequence[int]):
        """Init 2D conv kernel with kernel_size"""
        assert len(kernel_size) == 2
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        """Forward move in module"""
        return corr2d(X, self.weight) + self.bias

def corr_one_out(X, kernel):
    """Evaluate correlation on Muitiple input channels with one output channel."""
    return sum(corr2d(x, k) for x, k in zip(X, kernel))

def corr_multi(X, kernels):
    """Evaluate correlation on multiple in-out channels."""
    return torch.stack([corr_one_out(X, k) for k in kernels], 0)

def pool2D(X, pool_size: tuple[int, int], mode: str='max'):
    """Pooling on X.
    params:
        pool_size: size of pooling window
        mode: string of pooling mode (max or avg)
    """
    pool_h, pool_w = pool_size
    Y = torch.zeros((X.shape[0] - pool_h + 1, X.shape[1] - pool_w + 1))
    if mode == 'max':
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = X[i:i + pool_h, j:j + pool_w].max()
    elif mode == 'avg':
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = X[i:i + pool_h, j:j + pool_w].mean()
    return Y


if __name__ == '__main__':
    # Try to learn a kernel based on given input-output pair
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    Y = corr2d(X, torch.tensor([[1.0, -1.0]])).reshape((1, 1, 6, 7))
    X = X.reshape((1, 1, 6, 8))
    learning_rate = 3e-2
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # Iterate the kernel
        conv2d.weight.data[:] -= learning_rate * conv2d.weight.grad
        print(f"Epoch {i + 1}, loss {l.sum():.3f}")
