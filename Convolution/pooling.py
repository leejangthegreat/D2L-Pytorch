"""
Test for Pooling window evaluation.
"""
from time import perf_counter
import torch
from torch import nn
from d2l import torch as d2l
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

def test_pool2D(X, pool_size, mode='max'):
    """
    Test prof.
    """
    h, w = pool_size
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + h, j:j + w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + h, j:j + w].mean()
    return Y

if __name__ == '__main__':
    X = torch.rand((1000, 1000))
    pool_window = (2, 2)
    start1 = perf_counter()
    pool2D(X, pool_window)
    end1 = perf_counter()
    print(f"pool2D时间：{end1 - start1}")
    
    start2 = perf_counter()
    test_pool2D(X, pool_window)
    end2 = perf_counter()
    print(f"test2d: {end2 - start2}")

    start3 = perf_counter()
    pool2D(X, pool_window)
    end3 = perf_counter()

    print(f"pool2D-2：{end3 - start3}")

    start4 = perf_counter()
    test_pool2D(X, pool_window)
    end4 = perf_counter()
    print(f"test2D-2: {end4 - start4}")
