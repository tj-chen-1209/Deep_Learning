# # GG 没整明白 如何自己实现SoftMax函数 2025.7.17
# import torch
# from d2l import torch as d2l
# import matplotlib.pyplot as plt # 画图
# # import numpy as np 
# from IPython import display
# from d2l import torch as d2l

# # 1. 模型建立
# def SoftMax(X):
#     return torch.exp(X) / torch.exp(X).sum(axis=1, keepdim=True) # axis = 1 是按照axis=1的方向进行求和，keepdim=True保持维度不变 便于使用广播机制

# def Net(X, w, b):
#     """
#     定义一个简单的神经网络模型
#     但是由于输入是一个batch_size * 28 * 28的图像
#     所以需要将其展平为一个一维向量 
#     """
#     X = X.reshape(X.shape[0], -1)  # 展平每个样本 是一个batch_size * 784的矩阵
#     return SoftMax(torch.matmul(X, w) + b)  # 矩阵乘法得到每个样本的预测结果 
#     '''
#     得到的矩阵应该是一个batch_size * labels_num的矩阵
#     '''

# # 2. 交叉熵损失函数
# def Cross_Entropy(y_hat, y):
#     '''
#     负对数函数，但要注意将每一行即每个样本的预测结果取出来(最大值索引) y_hat[range(len(y_hat)), y]
#     '''
#     return -torch.log(y_hat[range(len(y_hat)), y])  # 主要是把y作为索引在y^里面取每一个数据
#     # return -torch.log(y_hat[:, y])  # 主要是把y作为索引在y^里面取每一个数据

# # 3. 准确率计算
# def Accuracy(y_hat, y):
#     '''
#     先从y_hat中取出每一行的最大值索引
#     数据类型变化后和y进行比较 比如 y_hat.type(y.dtype) == y
#     '''
#     if(len(y_hat.shape) > 1 and y_hat.shape[1] > 1): # 或者是 y_hat.shape[0] > 1 len()表示取第一维度
#         y_hat = y_hat.argmax(axis=1) # 取每一行的最大值索引 y_hat变成一个一维向量 维度是(batch_size,)
#     cmp = y_hat == y  # 这句话就是从维度(batch_size,)上一个一个找对应的labels，比如y_hat = [2, 1] y_hat.type(y.dtype)就是[[0, 0, 0, 1 ...],[0, 1 ,0, ...]]
#     return cmp.sum()  # 返回预测正确的数量

# # 4. 计算在指定数据集上的准确率
# def Evaluate_Accuracy(data_iter, net, w, b):
#     '''
#     计算在指定数据集上的准确率
#     data_iter: 数据迭代器
#     net: 模型
#     w: 权重
#     b: 偏置
#     '''
#     if isinstance(net, torch.nn.Module):
#         net.eval()  # 将模型设置为评估模式
#     acc_sum, n = 0.0, 0  # 初始化准确率和样本数
#     for X, y in data_iter:
#         acc_sum += Accuracy(net(X, w, b), y)  # 使用net计算预测结果
#         n += y.shape[0]  # 累计样本数量
#     return acc_sum / n  # 返回平均准确率

# # 5. 累加器类
# class Accumulator:  #@save
#     """在n个变量上累加"""
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# def train_epoch_ch3(net, train_iter, loss, updater):  #@save
#     """训练模型一个迭代周期（定义见第3章）"""
#     # 将模型设置为训练模式
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     # 训练损失总和、训练准确度总和、样本数
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         # 计算梯度并更新参数
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             # 使用PyTorch内置的优化器和损失函数
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             # 使用定制的优化器和损失函数
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), Accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]


# def main():
#     batch_size = 256
#     train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 维度是256*1*28*28 迭代器

#     num_inputs = 784 # 将28*28的图像展平为784维向量
#     num_outputs = 10

#     w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) # 初始化权重 高斯均值
#     b = torch.zeros((1, num_outputs), requires_grad=True) # 初始化权重和偏置 
#     X = torch.normal(0, 1, (2, 5))
#     Evaluate_Accuracy(Net(X, w, b), test_iter)



# if __name__ == "__main__":
#     main()

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from IPython import display
import matplotlib.pyplot as plt

# 数据集相关 --------------------------------------------------------------------------------------------------
# 加载数据集
def load_data_fashion_mnist(batch_size, num_workers=0, root='../data'):
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

# 数据集标签转换
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] # 将标签转换为对应的文本描述 比如输入的是[0, 1, 2] 就会返回 ['t-shirt', 'trouser', 'pullover']

# 显示数据图片
def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg') # 设置显示格式为矢量图
    
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy()) # 将图像展平为28x28的矩阵的numpy数组，以便调用matplotlib显示
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


# 模型定义 --------------------------------------------------------------------------------------------------------
def softmax(O):
    O_exp = O.exp()                             # 所有元素求 exp
    partition = O_exp.sum(dim=1, keepdim=True)  # 对列求和
    return O_exp / partition                    # 这里应用了广播机制

# 模型定义
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
# batch _size * 28 * 28 的图像展平为 batch_size * 784 的矩阵，W 是 784 * 10 的权重矩阵，b 是 1 * 10 的偏置向量
# 相当于对每个样本的 784 个特征进行线性变换，得到 10 个类别的预测分数，然后通过 softmax 函数将其转换为概率分布 离散来看就是每个标签都有一个w与b

# 交叉熵损失
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))  # 这里返回 n 个样本各自的损失，是 nx1 向量

# 优化方法：小批量随机梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size      # 注意这里更改 param 时用的param.data，这样不会影响梯度计算

# 准确率评估
def evaluate_accuracy(data_iter, net):
    acc_sum = 0.0  # 所有样本总准确率
    n =  0         # 总样本数量
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]
    return acc_sum / n

# 模型训练 --------------------------------------------------------------------------------------------------------
def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None):
    # 训练执行 num_epochs 轮
    for epoch in range(num_epochs):
        train_l_sum = 0.0    # 本 epoch 总损失
        train_acc_sum = 0.0  # 本 epoch 总准确率
        n = 0                # 本 epoch 总样本数
        
        # 逐小批次地遍历训练数据
        for X, y in train_iter: # 这里的 X 是一个 batch_size * 28 * 28 的张量，y 是一个 batch_size 的标签向量
            
            # 计算小批量损失
            y_hat = net(X)
            l = loss(y_hat, y).sum()  

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
        
            # 小批量的损失对模型参数求梯度
            l.backward()
            
            # 做小批量随机梯度下降进行优化
            sgd(params, lr, batch_size)   # 手动实现优化算法
 
            # 记录训练数据
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        
        # 训练完成一个 epoch 后，评估测试集上的准确率
        test_acc = evaluate_accuracy(test_iter, net)
        
        # 打印提示信息
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == '__main__':
    # 输入输出维度
    num_inputs,num_outputs = 28*28,10   # 图像尺寸28x28，拉平后向量长度为 28*28；类别空间为 10

    # 初始化模型参数 & 设定超参数
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True) 
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    num_epochs, lr = 5, 0.1             # 超参数

    # 获取数据读取迭代器
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, 4)

    # 进行训练
    train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    # 使用得到模型预测 10 张图
    X, y = iter(test_iter).next()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    # 显示预测结果
    show_fashion_mnist(X[0:9], titles[0:9])
    plt.show()
