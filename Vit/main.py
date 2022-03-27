import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

import utils
from model import VisionTransformer

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

batch_size = 8
class_num = 5
root = "./data/flower_photos"
train_images_path, train_images_label, val_images_path, val_images_label = utils.read_split_data(root)
train_data_set = utils.MyDataSet(images_path = train_images_path, image_class = train_images_label, transform = utils.data_transform["train"])
train_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
train_num = len(train_data_set)
validate_data_set = utils.MyDataSet(images_path = val_images_path, image_class = val_images_label, transform = utils.data_transform["val"])
validate_loader = DataLoader(validate_data_set, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)
val_num = len(validate_data_set)

print("using {} images for training, {} images for validation.".format(train_num, val_num))

model = VisionTransformer()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 100
best_acc = 0.0
save_path = './ViT.pth'
train_steps = len(train_loader)

print("Start training")
for epoch in range(epochs):
    # 冻结backbone网络，只训练classifier部分
    if epoch < 10:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    # 训练整个网络
    else:
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader):
        images, lables = data
        images, lables = images.to(device), lables.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, lables)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, running_loss / train_steps))

    model.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                predict_y = torch.max(outputs, dim = 1)[1]
  
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] val_accuracy: %.3f' %(epoch + 1, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

torch.cuda.empty_cache() # 清除部分无用变量
print('Finished Training')
