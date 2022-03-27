import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from model import Model

import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

batch_size = 64
class_num = 5
root_train = "./data/flower/flower_train"
root_val = "./data/flower/flower_val"

train_images_path, train_images_label = utils.read_split_data(root_train)
val_images_path, val_images_label = utils.read_split_data(root_val)
train_data_set = utils.MyDataSet(images_path = train_images_path, image_class = train_images_label, transform1 = utils.data_transform["train1"], transform2 = utils.data_transform["train2"])
train_loader = DataLoader(train_data_set, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
train_num = len(train_data_set)
memory_data_set = utils.MyDataSet(images_path = train_images_path, image_class = train_images_label, transform1 = utils.data_transform["val"], transform2 = utils.data_transform["val"])
memory_loader = DataLoader(memory_data_set, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)
validate_data_set = utils.MyDataSet(images_path = val_images_path, image_class = val_images_label, transform1 = utils.data_transform["val"], transform2 = utils.data_transform["val"])
validate_loader = DataLoader(validate_data_set, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)

val_num = len(validate_data_set)
print("using {} images for training, {} images for validation.".format(train_num, val_num))

epochs = 500
temperature = 0.5
best_acc = 0.0
save_path = './SimCLR.pth'
train_steps = len(train_loader)

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-6)

results = {'train_loss': [], 'val_acc': []}


for epoch in range(epochs):
    # 训练
    model.train()
    running_loss = 0.0
    for i, (imgs_1, imgs_2, labels) in enumerate(train_loader):
        imgs_1 = imgs_1.to(device)
        imgs_2 = imgs_2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # out_1=out2=[b,128]
        # 共享权重, 使用同一个网络进行输出
        feature_1, out_1 = model(imgs_1)
        feature_2, out_2 = model(imgs_2)
        # out=[2b,128]
        out = torch.cat([out_1, out_2], dim = 0)
        # sim_matrix=[2b,2b]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # mask=[2b,2b]，除了对角线为False，其余地方为True
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device = sim_matrix.device)).bool()
        # 删除掉对角线元素，用来做softmax的分母，因为对角线元素相当于自己与自己对比，sim_matrix=[2b,2b-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # 计算exp(si,sj)/temperature，si和sj是来自相同的图像的变换，img_sim_positive=[b]
        img_sim = torch.exp(torch.sum(out_1 * out_2, dim = -1) / temperature)
        # 在图像位置互换的情况下, 再次计算同一对图像的损失
        img_sim = torch.cat([img_sim, img_sim], dim = 0)
        # 损失函数
        loss = (-torch.log(img_sim / sim_matrix.sum(dim = -1))).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_loss = running_loss / train_steps
    results['train_loss'].append(train_loss)
    print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, train_loss))

    # 验证
    model.eval()
    acc = 0.0
    total_num = 0
    feature_bank = []
    labels_bank = []
    
    with torch.no_grad():
        # memory_loader返回两批相同的imgs
        for i, (imgs, _, labels) in enumerate(memory_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # feature=[b,feature_dim]
            feature, out = model(imgs)
            feature_bank.append(feature)
            labels_bank.append(labels)
        # 将验证集所有feature在0维上拼接起来，再进行转置，feature_bank=[feature_dim,N]
        feature_bank = torch.cat(feature_bank, dim = 0).t().contiguous()
        # feature_labels=[N]
        feature_labels = torch.cat(labels_bank, dim = 0).to(device)

        # 用验证集的feature与memory_loader的feature_bank做对比
        for i, (imgs, _, labels) in enumerate(validate_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # feature=[b,feature_dim]
            feature, out = model(imgs)
            total_num += imgs.size(0)   
            # sim_matrix=[b,N]
            sim_matrix = torch.mm(feature, feature_bank)
            # sim_indices=[b,1]
            _, sim_indices = sim_matrix.max(dim = -1)
            predicate_labels = feature_labels[sim_indices]
            acc += torch.eq(predicate_labels, labels).sum().item()
        val_accurate = acc / total_num
        print("val epoch[{}/{}] accuracy:{:.3f}".format(epoch + 1, epochs, val_accurate))
        results['val_acc'].append(val_accurate)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    data_frame = pd.DataFrame(data = results)
    data_frame.to_csv('./results/flower_500_64.csv'.format('{}_{}'), index_label='epoch')


print('Finished Training')