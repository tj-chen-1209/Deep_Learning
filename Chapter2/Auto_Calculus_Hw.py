import torch
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def df(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# 数据准备
x = np.linspace(0, 2 * np.pi, 100)
y = f(x)

# 图像1：使用接口
plt.figure(figsize=(12, 6))
plt.plot(x, y, label="Sin(x)")
plt.title("Using Interface: Sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend() # 显示图例 即上面的label

# 图像2：sin(x) 的数值导数
y_prime = df(f, x)

plt.plot(x, y_prime, label="Numerical Derivative of Sin(x)", color='r')
plt.title("Without Cos(x): Derivative of Sin(x)")
plt.xlabel("x")
plt.ylabel("y'")
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()

