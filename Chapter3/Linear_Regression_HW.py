''''# 实现一个随机梯度下降算法来拟合回归函数
import numpy as np
import torch

def f(x):
    """目标函数"""
    return 1 / 2 * np.exp(- abs(x))

# 提出一种随机梯度下降算法来解决这个问题。
def stochastic_gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    """随机梯度下降算法"""
    w = torch.randn(1, requires_grad=True)  # 初始化权重
    for epoch in range(epochs):
        for i in range(len(x)):
            xi = x[i].view(1, -1)  # 取出当前样本 对于n*1维度的向量并没啥用
            yi = y[i]  # 取出当前标签
            loss = (w * xi - yi).pow(2).mean()  # 计算损失 相当于1/n * (w * xi - yi)
            loss.backward()  # 反向传播
            with torch.no_grad(): # 禁用梯度计算
                w -= learning_rate * w.grad  # 更新权重
                w.grad.zero_()  # 清零梯度
    return w.item()

# 生成数据
def generate_data(num_samples=100):
    """生成数据"""
    x = torch.linspace(-5, 5, num_samples).view(-1, 1)  # 生成输入数据 生成列向量
    y = f(x.numpy()) + np.random.normal(0, 0.1, x.shape)  # 添加噪声 将torch张量转换为Numpy数组
    return x, torch.tensor(y, dtype=torch.float32) 

# 主函数
def main():
    """主函数"""
    x, y = generate_data()  # 生成数据
    w = stochastic_gradient_descent(x, y)  # 执行随机梯度下降
    print(f'拟合的权重: {w:.4f}')  # 输出拟合的权重

if __name__ == "__main__":
    main()
    '''

'''
问题很大！❌❌❌ 希望可以回来看 2025.7.15
'''

import numpy as np

# 目标函数 f(x) 及其梯度（关于 L(x) = -f(x) 的梯度，用于最小化 L）
def f(x):
    return 0.5 * np.exp(-np.abs(x))

def grad_L(x):
    """
    计算 L(x) = -f(x) 对 x 的导数
    f'(x) = -(1/2) * exp(-|x|) * sign(x)
    => L'(x) = -f'(x) =  0.5 * exp(-|x|) * sign(x)
    """
    return 0.5 * np.exp(-np.abs(x)) * np.sign(x)

# 超参数
lr = 0.1             # 初始学习率
n_iters = 1000       # 迭代次数
decay = 0.99         # 学习率衰减因子

# 随机初始化 x
x = np.random.randn()

# SGD 主循环
for i in range(n_iters):
    # “随机”——在这里就是每次都拿同一个函数，其实等价于普通梯度下降
    # 问题是在驻点处容易来回跳动 不会收敛到驻点
    g = grad_L(x)        # 计算梯度
    x -= lr * g          # 梯度下降一步
    lr *= decay          # 衰减学习率（帮助收敛）

# 输出结果
print(f"SGD 后的 x ≈ {x:.6f}")
print(f"f(x) ≈ {f(x):.6f}，理论最大值 f(0)=0.5")
