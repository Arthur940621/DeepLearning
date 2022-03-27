import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.utils.data import dataset
from torchvision import transforms
import torch.nn.functional as F
import load_minist_data

# 特征转换
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)) # 均值是0.1307，标准差是0.3081，这些系数都是数据集提供方计算好的数据
])

# 将数据集转换为Tensor的类
class DealDataset(Data.Dataset):
    def __init__(self, folder, data_name, label_name, transform = None):
        (dataSet, labels) = load_minist_data.load_data(folder, data_name, label_name)
        self.dataSet = dataSet
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataSet[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataSet)

# 设置文件路径
data_folder = "./Data/mnist"
train_data_name = "train-images-idx3-ubyte.gz"
train_label_name = "train-labels-idx1-ubyte.gz"
test_data_name = "t10k-images-idx3-ubyte.gz"
test_label_name = "t10k-labels-idx1-ubyte.gz"

# 实例化这个类，然后我们就得到了Dataset类型的数据
trainDataset = DealDataset(data_folder, train_data_name, train_label_name, transform)
testDataset = DealDataset(data_folder, test_data_name, test_label_name, transform)

# 训练数据和测试数据的装载
train_loader = Data.DataLoader(dataset = trainDataset, batch_size = 100, shuffle = True)
test_loader = Data.DataLoader(dataset = testDataset, batch_size = 100, shuffle = False)

# 构建模型
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # size的第一个维度是该批样本的数目，其他维度相乘为所有特征数目
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 实例化模型，并将模型加载到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet_5()
model.to(device)

# 构建损失函数核优化函数
criterion = nn.CrossEntropyLoss() # 交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

# 训练
def tarining(model, train_loader, criterion, optimizer, epochs = 1):
    sum_loss = 0.0
    for e in range(epochs):
        for i, data in enumerate(train_loader):
            (images, labels) = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' % (e + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
                
    print('finish training')

tarining(model, train_loader, criterion, optimizer, epochs = 10)

torch.save(model.state_dict(), 'LeNet_5.pth')