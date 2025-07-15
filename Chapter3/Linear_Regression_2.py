# 高斯分布的绘制
import numpy as np
import matplotlib.pyplot as plt

# ----------- 1. 定义正态分布函数 -----------
def normal(x, mu, sigma):
    """返回一维正态分布在 x 处的概率密度值"""
    coef = 1.0 / (np.sqrt(2 * np.pi) * sigma)        # 前面的系数
    expo = np.exp(-0.5 * ((x - mu) / sigma)**2)      # 指数部分
    return coef * expo

# ----------- 2. 生成横坐标 -----------
x = np.arange(-7, 7, 0.01)

# ----------- 3. 配置待绘制的 (均值, 标准差) 组合 -----------
params = [(0, 1), (0, 2), (3, 1)]

# ----------- 4. 绘图 -----------
plt.figure(figsize=(4.5, 2.5)) # 设置图形大小
for mu, sigma in params: # 遍历每个 (均值, 标准差) 组合
    plt.plot(x,
             normal(x, mu, sigma),
             label=f'mean {mu}, std {sigma}') # 绘制正态分布曲线

plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.grid(True)          # 网格可读性更好，可按需移除
plt.tight_layout()      # 紧缩布局，避免标签被裁剪
plt.show()

# 什么时候可能比使用随机梯度下降更好？这种方法何时会失效？ 
# 当数据集较小且可以在内存中处理时，批量梯度下降可能更好，因为它可以充分利用所有数据来计算梯度。
# 但是，当数据集非常大时，批量梯度下降可能会变得非常慢，并且需要大量的内存。

