from my_dataset import MyDataSet
from model import ResNet34
from utils import read_split_data

import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

root = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

data_transform = {
    "train":transforms.Compose([transforms.Resize([64, 64]),
                                transforms.RandomHorizontalFlip(), # 随机水平翻转
                                transforms.ToTensor(), # 转化为Tensor
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), # 标准化处理
   
    "val":transforms.Compose([transforms.Resize([64, 64]),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

batch_size = 32

train_data_set = MyDataSet(images_path = train_images_path, image_class = train_images_label, transform = data_transform["train"])
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = train_data_set.collate_fn)
train_num = len(train_data_set)
validate_dataset = MyDataSet(images_path = val_images_path, image_class = val_images_label, transform = data_transform["val"])
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = 0,  collate_fn = train_data_set.collate_fn)
val_num = len(validate_dataset)
print("using {} images for training, {} images for validation.".format(train_num, val_num))


model = ResNet34(6)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100
best_acc = 0.0
save_path = './ResNet34.pth'
train_steps = len(train_loader)

for epoch in range(epochs):
    # 训练
    running_loss = 0.0

    for step, data in enumerate(train_loader):
        images, lables = data
        images, lables = images.to(device), lables.to(device)
        optimizer.zero_grad()
        predict_y = model(images)
        loss = loss_fn(predict_y, lables)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("train epoch[%d/%d] loss:%.3f" % (epoch + 1, epochs, running_loss / train_steps))

    # 测试
    acc = 0.0
    with torch.no_grad():
        model.eval()
        for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                predict_y = torch.max(outputs, dim = 1)[1] # torch.max(outputs, dim = 1)返回每一行的最大值和索引，可以把索引看作为分类
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accurate = acc / val_num
        print('val epoch[%d/%d] val_accuracy: %.3f' % (epoch + 1, epochs, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
print('Finished Training')