import torch

''' 
在进行反向传播(backpropagation)时,能够计算梯度。
自动微分(autograd)是PyTorch的一个核心特性,它允许用户自动计算张量的梯度。当你对一个张量进行操作时,PyTorch会记录这些操作,并在需要时自动计算梯度。
一共三个步骤：
1. 创建一个张量并设置requires_grad=True,表示需要计算梯度。
2. 执行一些操作，生成一个新的张量。
3. 调用.backward()方法来计算梯度。
之后便可以通过访问张量的.grad属性来获取梯度。获得的梯度是相对于创建张量时的操作而言的。
比如,如果你创建了一个张量x,并对它进行了某些操作生成了一个新的张量y,那么调用y.backward()会计算y相对于x的梯度。并且x.grad将包含这个梯度的值。
下面的代码演示了如何使用PyTorch进行自动微分:'''

x = torch.arange(4.0)
x.requires_grad_(True)

y = 2 * torch.dot(x, x)
y.backward()  # Call backward to compute gradients
print(x.grad)  # Print the gradients

x.grad.zero_()  # ⭐清除梯度 如果不清除梯度，梯度会累加
y = sum(x)
y.backward()  # Recompute gradients
print(x.grad)  # Print the gradients again


'''
分离计算：如果你想要在计算图中断开某个张量的梯度传播，可以使用.detach()方法。这个方法会返回一个新的张量，它与原始张量共享数据，但不再需要梯度计算。
'''
# x.grad.zero_()
# y = x * x
# u = y.detach() # ⭐把y从计算图中分离出来 调用y.grad会报错 断开计算图 相当于在这之前的x对于后续的计算不再有影响 认为是常数
# z = u * x

# z.sum().backward()
# print(x.grad == u)

'''
HomeWork:
2. 在运行反向传播函数之后，立即再次运行它，看看会发生什么。
在同一个计算图上尝试进行了两次反向传播，而计算图在第一次反向传播后已经被释放。
如果你需要第二次反向传播，或者在反向传播后继续访问某些中间张量，你需要显式地保留计算图，使用 retain_graph=True 参数。
'''
