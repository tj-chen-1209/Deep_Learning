# from L_R_build_1 import synthetic_data, data_iter, linear_regression, squared_loss, sgd
import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

'''
    这个函数的作用是将特征和标签转换为PyTorch张量
'''
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # 将数据转换为TensorDataset 一个list
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 从TensorDataset中随机加载batch个数据 返回一个DataLoader对象

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 生成合成数据 调用d2l库中的synthetic_data函数
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)  # 使用load_array函数加载数据迭代器
    # Sequential是一个容器模块，可以将多个层按顺序组合起来 'List of layers'
    net = nn.Sequential(nn.Linear(2, 1))  # 定义线性回归模型 使用nn.Sequential将线性层封装起来
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()  # 定义损失函数 使用均方误差损失函数
    trainer = torch.optim.SGD(net.parameters(), lr=0.05)  # 定义优化器 使用随机梯度下降优化器
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y).mean()  # 计算损失函数的平均值
            trainer.zero_grad()
            l.backward() # 计算梯度
            trainer.step() # 更新参数
        l = loss(net(features), labels).mean()  # 计算整个数据集的损失函数的平均值
        print(f'epoch {epoch + 1}, loss {l:f}')
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)

if __name__ == '__main__':
    main()
    