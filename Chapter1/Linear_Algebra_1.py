import torch

# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())

# X = x.reshape(3, 4)
# print(X)

# y = torch.zeros(2, 3, 4)
# print(y)

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算

# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)) #dim=0表示行拼接，dim=1表示列拼接
# print(X == Y)  # 比较两个张量是否相等
# print(X.sum())  # 求和

a = torch.arange(6).reshape((3, 2))
# b = torch.arange(2).reshape((1, 2))
# print(a + b)  # 广播机制：a和b的形状不同，但可以通过广播机制进行相加 
# 广播机制：根据规则自动扩展形状，即复制行或列，使得两个张量的形状相同
# print(a[-1], a[1:3])  # 访问张量的最后一行和第2到第3行 
# a[1:3]表示包括第二行但不包括第三行的索引
# a[1,1] = 9  # 修改张量的值

# a[0:2, :] = 10  # 修改张量的第一行的所有列的值为10

# 为了减少内存开销：要使用索引操作 a[:] 或者 X[:] = X + Y ; X += Y
# before = id(a)
# a[:] = a + 2  # 使用切片操作修改张量的值
# after = id(a)
# print(before, after)  # 修改前后的id相同，说明没有创建新的张量

A = a.numpy() # 将张量转换为NumPy数组
B = torch.tensor(A) # 将NumPy数组转换为张量
print(type(A), type(B))


print(a)
