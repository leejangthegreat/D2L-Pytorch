"""
An implementation of batch norm layer in NN.
"""

import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    @params:
        gamma: scaling param
        beta: shifting param
    """
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else :
        assert len(X.shape) in  (2, 4)
        if len(X.shape) == 2:
            # For Linear Layers
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # For convolution layers, calc on axis 1
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # update moving mean and var
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    """
    Module implemented for batch norm between activation and linear or convolutional layer
    """
    def __init__(self, num_features: int, num_dims: int):
        super().__init__()
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma, self.beta = nn.Parameter(torch.ones(shape)), nn.Parameter(torch.zeros(shape))
        self.moving_mean, self.moving_var = torch.zeros(shape), torch.ones(shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Judge which device X is on
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.movng_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y
