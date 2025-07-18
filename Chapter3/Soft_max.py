# %matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images in a grid."""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # img: tensor of shape [1, H, W]
        img = img.squeeze(0)  # remove channel dim
        ax.imshow(img.numpy(), cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()] # 将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
    if resize: 
        trans.insert(0, transforms.Resize(resize)) 
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def main():
    # d2l.use_svg_display()
    # # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # # 并除以255使得所有像素的数值均在0～1之间
    # trans = transforms.ToTensor()
    # mnist_train = torchvision.datasets.FashionMNIST(
    #     root="../data", train=True, transform=trans, download=True)
    # mnist_test = torchvision.datasets.FashionMNIST(
    #     root="../data", train=False, transform=trans, download=True)

    # # print(len(mnist_train), len(mnist_test))
    # # print(mnist_train[0][0].shape) #像素为28*28的灰度图像，标签为0-9的数字
    # # loader = data.DataLoader(mnist_train, batch_size=18, shuffle=True)

    # # # Get one batch of images
    # # images, labels = next(iter(loader))

    # # # Display the images with their labels
    # # titles = get_fashion_mnist_labels(labels)
    # # show_images(images, num_rows=2, num_cols=9, titles=titles)
    # # plt.tight_layout()
    # # plt.show()
    # batch_size = 256
    # train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
    #                          num_workers=get_dataloader_workers())
    # timer = d2l.Timer()
    # for X, y in train_iter:
    #     continue
    # print(f'{timer.stop():.2f} sec')
    # TODO : BATCH的问题
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    timer = d2l.Timer()
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        # break
        continue
    print(f'{timer.stop():.2f} sec')


if __name__ == '__main__':
    main()