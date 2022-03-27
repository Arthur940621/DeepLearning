import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 构建模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.hidden_activation = nn.Sigmoid()
    def forward(self, x):
        output = self.hidden_activation(self.linear(x))
        return output

# 模型实例化
model = LogisticRegression()

# 构建损失函数
loss_fn = nn.BCELoss()

# 构建优化器
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# 数据
x1 = torch.unsqueeze(torch.linspace(-10, 10, 50), dim = 1)
x2 = torch.randn(50, 1) * 10
y = torch.cat((torch.ones(24), torch.zeros(26))).type(torch.float)
x = torch.cat((x1, x2), dim = 1)
y = torch.unsqueeze(y, dim = 1)
print(x)
dataset = Data.TensorDataset(x, y)
dataloader = Data.DataLoader(dataset = dataset, batch_size = 5, shuffle = True)

# 训练
for e in range(1000):
    epoch_loss = 0
    epoch_acc = 0
    for i, (x_train, y_train) in enumerate(dataloader):
        optimizer.zero_grad()
        out = model(x_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss += loss

        mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
        correct = (mask == y_train).sum()  # 计算正确预测的样本个数
        acc = correct.item() / x_train.size(0)  # 计算精度

    if (e + 1) == 1 or (e + 1) == 10 or (e + 1) == 50 or (e + 1) == 100 or (e + 1) == 1000:
        print("epoch：%d，loss：%f" % ((e + 1, epoch_loss)))
        print("acc：%f" % acc)

# 绘制分类器超平面
w0, w1 = model.linear.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(model.linear.bias.item())
plot_x = np.arange(-1, 1, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.plot(plot_x, plot_y)
plt.show()
