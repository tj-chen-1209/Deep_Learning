# 矢量化加速
import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1] #返回最近一次的时间

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n) 
timer = Timer() 
'''
    完成了对矩阵的加法运算
    具体来说是每一行的元素相加
    循环很慢
'''
for i in range(n): 
    c[i] = a[i] + b[i] 
print(f'{timer.stop():.5f} sec')

timer.start()
'''
    矢量化加速
    直接对两个向量进行加法运算
    速度快很多
'''
d = a + b
print(f'{timer.stop():.5f} sec')
