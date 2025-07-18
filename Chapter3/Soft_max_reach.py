import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l

# Softmax函数的实现
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 定义模型
def net(X, W, b):
    """定义模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # reshape用于将输入展平  获得每个样本的预测结果 维度是1*10

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) # 主要是把y作为索引在y^里面取每一个数据

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 如果y_hat是一个维度大于1, 列数大于1的张量
        y_hat = y_hat.argmax(axis=1)  # 取每一行的最大值索引
    cmp = (y_hat.type(y.dtype) == y)  # 将y_hat转换为y的类型进行比较
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量s

def evaluate_accuracy(data_iter, net, W, b):
    """计算在指定数据集上的准确率"""
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += accuracy(net(X, W, b), y)  # 使用net计算预测结果
        n += y.shape[0]  # 累计样本数量
    return acc_sum / n  # 返回平均准确率

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期(定义见第3章)"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def updater(W, b, lr, batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def main():
    '''
    获取Fashion-MNIST数据集并测试softmax函数和模型
    '''
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 维度是256*1*28*28 迭代器

    num_inputs = 784 # 将28*28的图像展平为784维向量
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) # 初始化权重 高斯均值
    b = torch.zeros((1, num_outputs), requires_grad=True) # 初始化权重和偏置 
    print(W.shape, len(b.shape), W.shape[1])


    # X = torch.normal(0, 1, (1, 784))
    # P0 = net(X, W, b)  # 测试模型输出


if __name__ == "__main__":
    main()