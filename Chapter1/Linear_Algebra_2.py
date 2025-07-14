import os
import torch

# 定义一个函数来删除包含最多 NaN 值的列 注意返回值
def Drop_Col(data):
    nan_counts = data.isna().sum()
# 找出包含最多 NaN 值的列
    column_to_drop = nan_counts.idxmax()
# 删除该列
    data_ = data.drop(columns=[column_to_drop])
    return data_



# 创建数据目录和文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建数据目录，如果不存在则创建
data_file = os.path.join('..', 'data', 'house_tiny.csv')    # 数据文件路径
with open(data_file, 'w') as f:     # 打开文件进行写入
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# 导入pandas库来处理数据
import pandas as pd

data = pd.read_csv(data_file)
print(data)
data = Drop_Col(data)  # 删除包含最多NaN值的列
print(data)
# inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] # 分离输入和输出
# inputs = inputs.fillna(inputs.mean()) # 填充缺失值 .mean()计算每列的均值除了NaN值 fillna()方法将NaN值替换为均值
# inputs = pd.get_dummies(inputs, dummy_na=True) # 将类别变量转换为虚拟变量（独热编码） dummy_na=True表示将NaN值也转换为一个虚拟变量
# print(inputs, outputs)

# X = torch.tensor(inputs.to_numpy())  # 将输入数据转换为张量
# Y = torch.tensor(outputs.to_numpy())  # 将输出数据转换为张量
# print(X, Y)

