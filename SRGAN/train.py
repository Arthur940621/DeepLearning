import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import *
from model import *

os.makedirs("imgs", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 60000
BATCH_SIZE = 1

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


dataset = MyDataset(root_dir="./data/DIV2K_train_HR")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(in_channels=3).to(DEVICE)
disc = Discriminator(in_channels=3).to(DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()

for epoch in range(NUM_EPOCHS):
    for idx, (low_res, high_res) in enumerate(loader):
        high_res = high_res.to(DEVICE)
        low_res = low_res.to(DEVICE)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch+1, NUM_EPOCHS, idx+1, len(loader), loss_disc.item(), gen_loss.item()))

    save_image(torch.cat([high_res*0.5+0.5, fake], dim=2), './imgs/Images{}.png'.format(epoch))

torch.save(gen.state_dict(), './generator.pth')
torch.save(disc.state_dict(), './discriminator.pth')