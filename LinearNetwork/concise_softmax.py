# A concise realization of softmax network using torch
import torch 
import numpy as np
from d2l import torch as d2l
from softmax import train
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Define the network
net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))  # Use Flatten to adjust the shape of network input

def init_weight(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

# 从交叉熵函数输入的每个元素减去最大元素，可以在不改变结果的情况下防止数值爆炸
# 注意此时分母可能向0下溢
# 我们可以在交叉熵损失中传递未规范化的预测，保留原有的softmax结果同时直接计算对数
loss = torch.nn.CrossEntropyLoss(reduction='none')

# Optimizer
optim = torch.optim.SGD(net.parameters(), lr=0.1)

num_epoches = 20

train(net, train_iter, test_iter, loss, num_epoches, optim)

