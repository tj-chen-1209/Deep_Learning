# import torch
# from torch import nn
# from d2l import torch as d2l

# class Animator:  #@save
#     """在动画中绘制数据"""
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear',
#                  fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
#                  figsize=(3.5, 2.5)):
#         # 增量地绘制多条线
#         if legend is None:
#             legend = []
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes, ]
#         # 使用lambda函数捕获参数
#         self.config_axes = lambda: d2l.set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts

#     def add(self, x, y):
#         # 向图表中添加多个数据点
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#             x = [x] * n
#         if not self.X:
#             self.X = [[] for _ in range(n)]
#         if not self.Y:
#             self.Y = [[] for _ in range(n)]
#         for i, (a, b) in enumerate(zip(x, y)):
#             if a is not None and b is not None:
#                 self.X[i].append(a)
#                 self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#             self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

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
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     # 返回训练损失和训练精度
#     return metric[0] / metric[2], metric[1] / metric[2]

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

# def accuracy(y_hat, y):  #@save
#     """计算预测正确的数量"""
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())

# def evaluate_accuracy(net, data_iter):  #@save
#     """计算在指定数据集上模型的精度"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()  # 将模型设置为评估模式
#     metric = Accumulator(2)  # 正确预测数、预测总数
#     with torch.no_grad():
#         for X, y in data_iter:
#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]


# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# # PyTorch不会隐式地调整输入的形状。因此，
# # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
#     """训练模型（定义见第3章）"""
#     animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
#                         legend=['train loss', 'train acc', 'test acc'])
#     for epoch in range(num_epochs):
#         train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         animator.add(epoch + 1, train_metrics + (test_acc,))
#     train_loss, train_acc = train_metrics
#     assert train_loss < 0.5, train_loss
#     assert train_acc <= 1 and train_acc > 0.7, train_acc
#     assert test_acc <= 1 and test_acc > 0.7, test_acc


# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)

# net.apply(init_weights)
# loss = nn.CrossEntropyLoss(reduction='none')
# trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# num_epochs = 10
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# softmax_fashion_mnist.py
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_data_fashion_mnist(batch_size, root='../data'):
    """返回训练集和测试集 DataLoader"""
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=root, train=True,
                                     transform=transform, download=True)
    test_ds  = datasets.FashionMNIST(root=root, train=False,
                                     transform=transform, download=True)
    train_iter = data.DataLoader(train_ds, batch_size=batch_size,
                                 shuffle=True,  num_workers=4, pin_memory=True)
    test_iter  = data.DataLoader(test_ds,  batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)
    return train_iter, test_iter

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
class Accumulator:
    """在 n 个变量上累加"""
    def __init__(self, n): self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self): self.data = [0.0] * len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def accuracy(y_hat, y):
    """返回预测正确样本数"""
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat == y).sum())

@torch.no_grad()
def evaluate_accuracy(net, data_iter, device):
    """在 data_iter 上评估准确率"""
    net.eval()
    metric = Accumulator(2)          # 正确数、总数
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss_fn, optimizer, device):
    """单个 epoch 的训练过程，返回平均损失和准确率"""
    net.train()
    metric = Accumulator(3)          # 总损失、正确数、样本数
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric.add(loss.item() * y.numel(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss_fn, num_epochs, optimizer, device):
    """完整训练循环，并在训练结束后用 matplotlib 画图"""
    # 用列表收集指标
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(net, train_iter, loss_fn, optimizer, device)
        te_acc = evaluate_accuracy(net, test_iter, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(f"epoch {epoch:2d}: "
              f"train loss {tr_loss:.4f}, train acc {tr_acc:.3f}, test acc {te_acc:.3f}")

    # ------------------------- 训练结束，开始画图 -------------------------
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs,  label='train acc')
    plt.plot(epochs, test_accs,   label='test acc')
    plt.xlabel('epoch'); plt.ylabel('value'); plt.title('Training curves')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 仅包含 Flatten + 线性层的 Softmax 回归模型
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10)
    )

    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
    net.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train(net, train_iter, test_iter, loss_fn, num_epochs, optimizer, device)
