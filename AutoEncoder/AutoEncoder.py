import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.utils.data import dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import gzip
import numpy as np

# 模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # [b, 748] ==> [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # [b, 20] ==> [b, 768]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        e = self.encoder(x)
        e = F.dropout(e, 0.2)
        d = self.decoder(e)
        return e, d

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

transform = transforms.Compose(
    [
        transforms.ToTensor()
])

def load_data(data_folder, data_name, label_name):
    """
        data_folder:文件目录
        data_name:数据文件名
        label_name:标签数据文件名
    """
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y = np.frombuffer(lbpath.read(), np.uint8, offset = 8)
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y), 28, 28)
    return (x, y)

class DealDataset(Data.Dataset):
    def __init__(self, folder, data_name, label_name, transform = None):
        (dataSet, labels) = load_data(folder, data_name, label_name)
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

data_folder = "./Data/mnist"
train_data_name = "train-images-idx3-ubyte.gz"
train_label_name = "train-labels-idx1-ubyte.gz"
test_data_name = "t10k-images-idx3-ubyte.gz"
test_label_name = "t10k-labels-idx1-ubyte.gz"

trainDataset = DealDataset(data_folder, train_data_name, train_label_name, transform)
testDataset = DealDataset(data_folder, test_data_name, test_label_name, transform)
train_loader = Data.DataLoader(dataset = trainDataset, batch_size = 100, shuffle = True)
test_loader = Data.DataLoader(dataset = testDataset, batch_size = 100, shuffle = False)

# 构建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


for epoch in range(5):
    for i, data in enumerate(train_loader):
        images, _ = data
        images = images.to(device)

        optimizer.zero_grad()
        e, d= model(images)
        images = images.view(-1, 1 * 28 *28)
        loss = criterion(d, images)
        loss.backward()
        optimizer.step()
    print(epoch, 'loss', loss.item())


torch.save(model.state_dict(), "AutoEncoder.plk")

# 加载模型
model = AutoEncoder()
model.load_state_dict(torch.load("AutoEncoder.plk"))

for i, data in enumerate(test_loader):
    images, labels = data
    e_outpute, d_output = model(images)
    d_output = d_output.view(-1, 1, 28, 28)
print(images[1])

