import torch

# #对于一维张量
# a = torch.arange(12)  # 创建一个一维张量
# print(a.shape)  # 输出张量的形状
# print(len(a))  # 输出张量的长度

# b = torch.arange(12).reshape((3, 4))  # 创建一个二维张量
# print(b)  # 输出张量
# b = b.T  # 转置张量
# print(b)  # 输出转置后的张量

# B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(B.T == B) # 检查转置后的张量是否等于原张量

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
# print(A, A + B)

# A_sum_axis0 = A.sum(axis=0) # 沿着第0轴（行）求和 
# A_sum_axis1 = A.sum(axis=1) # 沿着第1轴（列）求和
# print(A_sum_axis0, A_sum_axis1)
# print(A.sum(axis=1, keepdims=True))  # 保持维度不变，返回一个列向量

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
# print(id(A) == id(B))  # 打印A和B的内存地址
# print(A, A + B, A * B)  # 打印A和B的值以及它们的和与积
# print(A.numel())

# A_sum_axis0 = A.sum(axis=0)  # 沿着第0轴（行）求和
# A_sum_axis0_re = A.sum(axis=0, keepdims=True)  # 保持维度不变，返回一个行向量 为了更好的使用广播机制
# print(A / A_sum_axis0, A / A_sum_axis0_re)  # 每个元素除以对应列的和 两种方式对于二维矩阵结果相同

# ✨ ⭐ ✅ ❌ 👀但是对于三维矩阵结果不同 : 比如 2*2*5的矩阵 按axis=1求和后，结果是2*5的矩阵 会在广播机制下变成2*5*1 而保持维度不变后，结果是2*1*5的矩阵 
# A = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5)
# A_sum_axis1 = A.sum(axis=1)  # 沿着第0轴（行）求和
# print(A_sum_axis1)  # 输出沿着第1轴求和后的结果
# A_sum_axis1_re = A.sum(axis=1, keepdims=True)  # 保持维度不变，返回一个行向量
# print(A / A_sum_axis1)  # 每个元素除以对应列的和
# print(A / A_sum_axis1_re)  # 每个元素除以对应列的和 保持维度不变

# print(A.cumsum(axis=0)) # 累加和
# print(A_sum_axis0.shape, A_sum_axis0_re.shape)

# 👀 点积： 向量间的点积 也可以认为是两个向量形成的平行四边形的面积
# A = torch.ones(4, dtype=torch.float32)  # 创建一个一维张量
# B = torch.arange(4, dtype=torch.float32)  # 创建另一个一维张量
# print(B)
# print(torch.dot(A, B))  # 矩阵乘法

# 👀 矩阵-向量积 若维度不变，则认为是对向量的旋转变换； 若维度变化，则认为是对向量的升维或降维
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个二维张量
# B = torch.arange(4, dtype=torch.float32)  # 创建一个一维张量
# print(torch.mv(A, B))  # 矩阵-向量积

# 👀 矩阵-矩阵积 认为是矩阵B原基底在矩阵A线性变换后新基底的新的矩阵表示
# ❌ 注意： 与hamilton积不同，矩阵-向量积的结果是一个向量，而不是一个矩阵 Hamilton积是矩阵对应元素的乘积，结果还是矩阵，或者是广播条件下的矩阵结果
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个二维张量
# B = torch.arange(20, dtype=torch.float32).reshape(4, 5)  # 创建另一个二维张量
# print(torch.mm(A, B))  # 矩阵-矩阵积

# 📔 范数：
# 向量的范数是向量长度的度量 L1：
# A = torch.arange(4, dtype=torch.float32)  # 创建一个一维张量
# print(torch.norm(A, p=1))  # L1范数
# # L2：
# print(torch.norm(A, p=2))  # L2范数
# # L∞：
# print(torch.norm(A, p=float('inf')))  # L∞范数
# # 矩阵的范数是矩阵的大小的度量
# # Frobenius范数：
# print(torch.norm(A.reshape(2, 2), p='fro'))  # Frobenius范数

# ✅ Homework:
# 1. 使用张量的转置和转置再转置的性质，验证一个张量是否等于其转置再转置的结果。
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个二维张量
# print(A == A.T.T) # 检查A的转置再转置是否等于A
# 2. 转置的加法性质：验证两个张量的转置相加是否等于它们的和的转置。
# A = torch.randn(3, 4)  # 创建一个3行4列的随机张量
# B = torch.randn(3, 4)  
# print((A.T + B.T) == (A + B).T)  # 输出A和B的转置相加的结果
# 3. 验证一个张量的转置加上其转置是否等于其转置。
# n = 7
# A = torch.randn(n, n) # 创建一个n行n列的随机张量
# print((A + A.T) == (A + A.T).T)
# 4. 验证一个三维张量的长度。是指张量的第一个维度的长度。
# A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # 创建一个三维张量
# print(len(A)) # 输出张量的长度
# 5. 对于任意形状的张量X,len(X)是否总是对应于X特定轴的长度?这个轴是什么? 
# 答案是：len(X)总是对应于X的第一个轴的长度。
# 6. 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因？ 详见✨注释部分
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 创建一个二维张量
# print(A / A.sum(axis=1))  # 每个元素除以对应行的和
# 7. 考虑一个具有形状(2, 3, 4)的张量，在轴0、1、2上的求和输出是什么形状?
# A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # 创建一个三维张量
# print(A.sum(axis=0).shape)  # 输出沿着第0轴求和后的形状 相当于两个平面矩阵对应的数据相加
# print(A.sum(axis=1).shape)  # 输出沿着第1轴求和后的形状 相当于每个平面上的行对应的数据相加
# print(A.sum(axis=2).shape)  # 输出沿着第2轴求和后的形状 相当于每个平面上的列对应的数据相加
# 8. 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?
# A = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # 创建一个三维张量
# print(torch.linalg.norm(A))  # 计算张量的L2范数 是所有元素的平方和的平方根

