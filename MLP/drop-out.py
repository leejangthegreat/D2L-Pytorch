import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch import nn
from typing import *

Metric = TypeVar('Metric')

def dropout_layer(X: Metric, p: float) -> Metric:
    assert 0 <= p <= 1
    if p == 1:
        return torch.zeros_like(X)
    if p == 0:
        return X
    else:
        dropout_mask = (torch.rand(X.shape) > p).float()
        return dropout_mask * X / (1 - p)  # Keep unbiased

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5  # 通常在靠近输入层的位置设置较低暂退概率

# Define Module
class Net(nn.Module):
    """Self-defined module with drop-out"""
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, drop1, drop2, is_training=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.ReLU = nn.ReLU()
        self.dropout1 = drop1
        self.dropout2 = drop2

    def forward(self, X):
        H1 = self.ReLU(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)

        H2 = self.ReLU(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)

        out = self.lin3(H2)
        return out
    def evaluate_loss(self, data_iter: Iterable):
        metric = d2l.Accumulator(2)
        self.eval()
        with torch.no_grad():
            for X, y in data_iter:
                out = self.forward(X)
                if len(out.shape) > 1 and out.shape[1] > 1:
                    out = out.argmax(axis=1)
                
                acc = float((out.type(y.dtype) == y).type(y.dtype).sum())
                metric.add(acc, y.numel())

        return metric[0] / metric[1]


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2, True)

num_epochs, lr, batch_size = 25, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)

animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                       xlim=[1, num_epochs], ylim=[0, 1], legend=['train', 'test'])

for epoch in range(num_epochs):
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
    animator.add(epoch + 1, train_metrics + (net.evaluate_loss(test_iter),))
    plt.pause(0.1)
