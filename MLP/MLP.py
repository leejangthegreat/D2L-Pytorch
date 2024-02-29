import torch 
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, sys.path[0] + '/../')
from Utils.Animator import Animator
from Utils.Accumulator import Accumulator
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# MLP Net
# MNIST每个图像有784个像素，共包含十个类别
num_inputs, num_outputs = 784, 10
# 通常隐藏层设置为2的幂次，在计算上更高效
num_hidden = 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hidden, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# Activation
def ReLU(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = ReLU(X@W1 + b1)
    return (H@W2 + b2)

# Loss
loss = nn.CrossEntropyLoss(reduction='none')

num_epoches, lr = 20, 0.1
updater = torch.optim.SGD(params, lr=lr)

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

def train_one_epoch(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
        net.train()

    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epoches, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epoches], ylim=[0, 1.0], 
                       legend=['train loss', 'train acc', 'test acc'])
    print(f"Training for {num_epoches} epoches.")

    for epoch in range(num_epoches):
        train_metric = train_one_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))

if __name__ == '__main__':
    train(net, train_iter, test_iter, loss, num_epoches, updater)
