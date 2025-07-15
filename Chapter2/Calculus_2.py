import matplotlib.pyplot as plt
import numpy as np

# Homework: 梯度求解
 
def f(x1, x2):
    return 3 * x1 **2 + 5 * np.exp(x2) # 调用numpy的指数函数

def f_L2(x):
    return np.linalg.norm(x)  # 二范数函数的梯度求解

# 数值方法求解 适用于多维函数的梯度 但是结果的精度较低 泛用性强 但三角函数等特殊函数可能不适用
def Numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x, y)) / h  # 对x的偏导数
    df_dy = (f(x, y + h) - f(x, y)) / h  # 对y的偏导数
    return np.array([df_dx, df_dy])

# L2范数的梯度求解 
def Analytical_gradient_L2(x):
    return x / np.linalg.norm(x)  # L2范数的梯度是单位向量

# 解析方法求解 适用于多维函数的梯度 但是需要手动计算偏导数 泛用性低但精度高 
def Analytical_gradient(x1, x2):
    df_dx_ = 6 * x1  # 对x1的偏导数
    df_dy_ = 5 * np.exp(x2)  # 对x2的偏导数
    return np.array([df_dx_, df_dy_])

x = 1.0
y = 1.0
print("f(x, y) =", f(x, y))
grad = Numerical_gradient(f, x, y)
print("Gradient at (x, y) = (1.0, 1.0):", grad)
print("Analytical gradient at (x, y) = (1.0, 1.0):", Analytical_gradient(x, y))

x = np.array([1.0, 1.0])
print("\nf_L2(x) =", f_L2(x))
print(Analytical_gradient_L2(x))

#f(x, y) = 16.591409142295227
# Gradient at (x, y) = (1.0, 1.0): [ 6.00003   13.5914771]
# Analytical gradient at (x, y) = (1.0, 1.0): [ 6.         13.59140914]
# 发现数值方法和解析方法的结果非常接近，说明数值方法的精度较高。  
# 但是数值方法的精度受h的影响较大，h越小，精度越高，但计算时间也越长。解析方法的精度较高，但需要手动计算偏导数，适用于简单函数的梯度求解。