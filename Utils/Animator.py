from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt
from typing import *
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel: Optional[str]=None, ylabel: Optional[str]=None, legend: Optional[List[str]]=None,
                 xlim: Optional[List[Union[int, float]]]=None, ylim: Optional[List[Union[int, float]]]=None,
                 xscale: Optional[str]='linear', yscale: Optional[str]='linear',
                 fmts: Optional[Tuple[str]]=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize: Optional[Tuple[Union[int, float]]]=(3.5, 2.5)):
        map(plt.close, plt.get_fignums())
        plt.cla()
        plt.ioff()

        plt.ion()
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
    
        display.display(self.fig)
        plt.pause(0.1)
        display.clear_output(wait=True)

