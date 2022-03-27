import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from model import *
from math import log2

os.makedirs("imgs", exist_ok=True)

Z_DIM = 512
IN_CHANNELS = 256
CHANNELS_IMG = 3
START_TRAIN_AT_IMG_SIZE = 128
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
LAMBDA_GP = 10 # 梯度惩罚损失权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "./data"

# 求惩罚梯度
def gradient_penalty(critic, real, fake, alpha, train_step):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(DEVICE)
    # 得到real和fake之间的样本
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, alpha, train_step)

    # 对interpolated_images求梯度
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# 读取数据
def get_loader(image_size):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = BATCH_SIZES[int(log2(image_size / 4))] # image_size越大，bitchsize越小
    dataset = datasets.ImageFolder(root=ROOT, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return dataloader, dataset

gen = Generator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

# 自动混合精度
scaler_critic = torch.cuda.amp.GradScaler()
scaler_gen = torch.cuda.amp.GradScaler()

gen.train()
critic.train()

step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))

for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-5
    loader, dataset = get_loader(4 * 2 ** step)
    print("Current image size: {}".format(4 * 2 ** step))

    for epoch in range(num_epochs):
        
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)

            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                critic_real = critic(real, alpha, step)
                critic_fake = critic(fake.detach(), alpha, step)
                gp = gradient_penalty(critic, real, fake, alpha, step)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

            with torch.cuda.amp.autocast():
                gen_fake = critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
            alpha = min(alpha, 1)

            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch+1, num_epochs, batch_idx+1, len(loader), loss_critic.item(), loss_gen.item()))

    step += 1 # 生成下一组图片尺寸
    save_image(fake, './imgs/FakeImages_{}_size.png'.format(4 * 2 ** step))

torch.save(gen.state_dict(), './generator.pth')
torch.save(critic.state_dict(), './discriminator.pth')