# matplotlib inline
import random
import torch
import matplotlib.pyplot as plt

'''
    重要！
    本代码简单实现了线性回归模型的训练过程，包括数据生成、数据迭代、模型定义、损失函数计算和参数更新。
    主要步骤如下：
    1. 生成合成数据：使用正态分布生成特征和标签。
    2. 数据迭代：将数据分批次处理，使用生成器函数返回每个批次的特征和标签。
    3. 定义线性回归模型：使用矩阵乘法和偏置项进行线性变换。
    4. 定义损失函数：使用平方损失函数计算预测值和真实标签之间的差异。
    5. 实现随机梯度下降优化器：更新模型参数，清空梯度。
    6. 训练模型：迭代多个周期，计算每个批次的损失，更新模型参数，并打印训练损失。
'''

def synthetic_data(w, b, num_examples): 
    '''
    生成正态分布数据 0是均值 1是标准差 后面的两个参数是样本数量和特征数量即张量的形状
    '''
    x = torch.normal(0, 1, (num_examples, len(w)))  # 生成正态分布数据 其中 num_examples 是样本数量，len(w) 是特征数量
    y = torch.matmul(x, w) + b  # 线性变换 根据不同的w和b生成标签 比如 w = [2, -3.4] 和 b = 4.2 时，y = 2*x1 - 3.4*x2 + 4.2
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return x, y.reshape((-1, 1))  # 返回特征和标签

def data_iter(batch_size, features, labels):
    num_examples = len(features) # 获取样本数量
    indices = list(range(num_examples)) # 获取样本索引列表 转换成以num_examples为长度的列表 为了随机读取样本
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices) # 打乱样本索引列表
    for i in range(0, num_examples, batch_size): # 按照 batch_size 划分批次 从0开始每次增加batch_size到 num_examples
        batch_real = min(i + batch_size, num_examples) - i # 计算当前批次的实际样本数量
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]) # 获取当前批次的样本索引  获取从 i 到 min(i + batch_size, num_examples) 的索引转换成张量
        yield features[batch_indices], labels[batch_indices], batch_real# 返回当前批次的特征和标签
'''
    这句话的意思是 从features中获取batch_indices向量位置的特征和标签 :
    比如batch_indices是[0, 1, 2],那么就返回features[0],features[1],features[2]对应的特征和标签
    yield 是一个生成器函数，它会返回一个迭代器，每次调用时返回当前批次的特征和标签。
'''

def linear_regression(X, w, b):
    return torch.matmul(X, w) + b  # 线性回归模型的前向传播

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 平方损失函数

def sgd(params, lr, batch_size):
    with torch.no_grad(): # 禁用梯度计算 这句话的作用是为了在更新参数时不计算梯度,节省内存以及加速计算
        # 遍历参数列表 params 中的每个参数 w, b 等
        for param in params:
            # param.grad 是参数的梯度，lr 是学习率，batch_size 是批次大小
            param -= lr * param.grad / batch_size # 更新参数 为什么要除以batch_size呢？因为获得的是整个batchsize的梯度和，要除以batch_size才能得到平均梯度
            param.grad.zero_() # 清空梯度
''' 随机梯度下降优化器
    params 是模型参数 lr 是学习率 batch_size 是批次大小
    这个函数的作用是更新模型参数
'''

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # print('features:', features[0],'\nlabel:', labels[0])
    batch_size = 10
    # for x, y in data_iter(batch_size, features, labels):
    #     print(x, '\n', y)
    #     break

    '''
    训练模型
    '''
    # w = torch.zeros(2, 1, requires_grad=True) # 初始化权重为0 需要梯度计算
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) # 初始化权重为正态分布 0是均值 0.01是标准差 size=(2,1)是形状 需要梯度计算
    b = torch.zeros(1, requires_grad=True) # 初始化偏置为0 需要梯度计算
    lr = 0.03 # 学习率
    num_epochs = 3 # 训练轮数
    net = linear_regression # 定义线性回归模型 这个意思是将net与linear_regression函数关联起来
    loss = squared_loss # 定义损失函数 这个意思是将loss与squared_loss函数关联起来

    for epoch in range(num_epochs): 
        for X, y, batch in data_iter(batch_size, features, labels): 
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯三
            l.sum().backward() # 计算梯度 sum是为了将l的所有元素加起来，得到一个标量，这样才能计算梯度
            sgd([w, b], lr, batch)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')

    # 只运行一次，打印出第一个批次的特征和标签


    # # 设置图形大小
    # plt.figure(figsize=(6, 4))  # 6 是宽度，4 是高度
    # # 绘制散点图
    # plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)  # s=1 表示点的大小
    # # 显示图形
    # plt.show()


if __name__ == "__main__":
    main()

