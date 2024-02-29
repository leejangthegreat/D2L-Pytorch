# Classification and Softmax network

import torch
from d2l import torch as d2l
from IPython import display
from typing import *
import sys
sys.path.insert(0, sys.path[0] + '/../')
from Utils.Accumulator import Accumulator
from Utils.Animator import Animator

# We will use MNIST dataset to show softmax classification
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Basic Model Params
num_inputs, num_outputs = 784, 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# Define Softmax Function
def softmax(X: torch.Tensor):
    X_exp = torch.exp(X)
    exp_sum = X_exp.sum(1, keepdim=True)
    return X_exp / exp_sum  # Broadcasting 

def net(X: torch.Tensor):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# Define Cross-Entropy Loss
# y is a one-hot label, choose one possibility in y_hat from all estimated possibility of every class
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# Classification Accuracy
def accuracy(y_hat, y):
    """Amount of correct classification"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # Use max p of every instance as result
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """Evaluate accuracy on specific data set"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# Train Process
def train_one_epoch(net, train_iter, loss, updater):
    """Train net for one epoch"""
    if isinstance(net, torch.nn.Module):
        net.train()  # Set train model
    metric = Accumulator(3)  # Total training loss, total training accuracy, number of instances

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()  # Use mean to present array of elements
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]

num_epoches = 10

def train(net, train_iter, test_iter, loss, num_epoch, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epoch], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    print(f"Training for {num_epoch} epoches.")
    for epoch in range(num_epoch):
        train_metric = train_one_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        animator.add(epoch + 1, train_metric + (test_acc,))
    train_loss, train_acc = train_metric  # Final result
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

if __name__ == '__main__':
    train(net, train_iter, test_iter, cross_entropy, num_epoches, updater)

