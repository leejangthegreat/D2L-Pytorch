import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

AlexNet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),  # 54x54
    nn.MaxPool2d(kernel_size=3, stride=2),  # 26x26
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),  # 26x26
    nn.MaxPool2d(kernel_size=3, stride=2),  # 12x12
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),  # 5x5
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)
AlexNet28 = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=5, padding=2), nn.ReLU(),  # 28x28
    nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),  # 14x14
    nn.MaxPool2d(kernel_size=3, stride=1), # 12x12
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),  # 12x12
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),  # 5x5
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)
if __name__ == '__main__':
    batch_size = 128
    trainIter, testIter = d2l.load_data_fashion_mnist(batch_size, resize=224)  # Instead of ImageNet, we use Fashion-MNIST here
    trainIter28, testIter28 = d2l.load_data_fashion_mnist(batch_size)
    learning_rate, num_epochs = 0.01, 10
    def evaluate_accuracy(net, data_iter):
        """Evaluate accuracy without GPU"""
        if isinstance(net, nn.Module):
            net.eval()

        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(d2l.accuracy(net(X), y), y.numel())

        return metric[0] / metric[1]

    def train(net, train_iter, test_iter, num_epoches, lr):
        """Train LeNet"""
        def init_weight(m):
            """Xavier initialization"""
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        net.apply(init_weight)

        optimizer = torch.optim.SGD(net.parameters(), lr = lr)
        loss = nn.CrossEntropyLoss()

        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epoches],
                            legend=['train loss', 'train acc', 'test acc'])

        timer, num_batches, examples = d2l.Timer(), len(train_iter), 0
        global_train_loss, global_train_acc, global_test_acc = 0, 0, 0

        for epoch in range(num_epoches):
            # 训练损失，准确度和样本数
            metric = d2l.Accumulator(3)
            net.train()

            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()

                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()

                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                examples += metric[2]
                global_train_acc, global_train_loss = train_acc, train_loss
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
                    plt.pause(0.1)

            test_acc = evaluate_accuracy(net, test_iter)
            global_test_acc = test_acc
            animator.add(epoch + 1, (None, None, test_acc))
            plt.pause(0.1)
        print(f'loss {global_train_loss:.3f}, train acc {global_train_acc:.3f}, '
              f'test acc {global_test_acc:.3f}')
        print(f'{examples / timer.sum():.1f} examples/sec')


    # train(AlexNet, trainIter, testIter, num_epochs, learning_rate)
    train(AlexNet28, trainIter28, testIter28, num_epochs, learning_rate)
