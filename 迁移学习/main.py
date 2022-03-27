import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchvision import models
import utils
import torch.optim as optim

batch_size = 128
root = "./Data/dogs-vs-cats"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

train_images_path, train_images_label, val_images_path, val_images_label = utils.read_split_data(root)

train_data_set = utils.MyDataSet(images_path = train_images_path, image_class = train_images_label, transform = utils.data_transform["train"])
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size = batch_size, shuffle = True, num_workers = 0)
train_num = len(train_data_set)
validate_dataset = utils.MyDataSet(images_path = val_images_path, image_class = val_images_label, transform = utils.data_transform["val"])
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
val_num = len(validate_dataset)
print("using {} images for training, {} images for validation.".format(train_num, val_num))

# 设定模型为vgg16，预训练为True
model = models.vgg16(pretrained=True)
# 先冻结全部参数，即使发生新的训练也不会进行参数的更新
for param in model.parameters():
    param.requires_grad = False
# 设置新的分类器
model.classifier[-1] = nn.Sequential(nn.Linear(in_features=4096, out_features=2, bias=True),
                                     nn.LogSoftmax(dim=1))

for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# 随着epoch的增大而逐渐减小学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
loss_fn = nn.NLLLoss()

epochs = 10
best_acc = 0.0
save_path = './VGG16.pth'
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
                predict_y = torch.max(outputs, dim = 1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
        val_accurate = acc / val_num
        print('val epoch[%d/%d] val_accuracy: %.3f' % (epoch + 1, epochs, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
    print('epoch=%d lr=%f' % (epoch + 1, scheduler.get_last_lr()[0]))
    scheduler.step()
print('Finished Training')