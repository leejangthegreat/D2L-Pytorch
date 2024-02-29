# 利用框架简洁实现 LN
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# Generate DataSet
true_w, true_b = torch.tensor([2, -3.4]), 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, shuffle=True):
    """A Pytorch Data Iter.
    :param data_arrays: tuple of features and labels
    :param  batch_size: size of a data batch
    :param     shuffle: whether data need shuffling
    :return           : the data_iter
    """
    dataSet = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataSet, batch_size, shuffle=shuffle)

# Read the data set
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# Build the Net (Use Sequential to connect all layers)
net = torch.nn.Sequential(torch.nn.Linear(2, 1))  # Input shape and output shape

# Initialize params in the 0th layer
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# Define loss function
loss = torch.nn.HuberLoss()

# Define optimizer
optim = torch.optim.SGD(net.parameters(), lr=0.03)

# Training Epoches
num_epoches = 3
for epoch in range(num_epoches):
    for X, y in data_iter:
        l = loss(net(X), y)
        optim.zero_grad()
        l.backward()
        optim.step()
    l = loss(net(features), labels)
    print(f"Epoch {epoch + 1}: loss {l:f}")

print('w的估计误差：', true_w - net[0].weight.data.reshape(true_w.shape))
print('b的估计误差：', true_b - net[0].bias.data)
