import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """Generate testing data with Gauss noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

def dataIter(batch_size, features, labels):
    """Generate batches from features and labels"""
    num_examples = len(features)
    indices = list(range(num_examples))

    # randomly get batch
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:max(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# Linear Network Components
def regFunc(X, w, b):
    return torch.matmul(X, w) + b

def MSE(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# Optimization Alg
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。由于稍后loss为一个batch loss的总和，梯度也为其和，故此处除以batch size以规范化步长。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# Initialize Parameters
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Training with initial params
lr = 0.03
num_epoches = 3
batch_size = 10
net = regFunc
loss = MSE

for epoch in range(num_epoches):
    for X, y in dataIter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_loss = loss(net(features, w, b), labels)
        print(f"Epoch {epoch + 1}: train loss = {float(train_loss.mean()):f}")

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
