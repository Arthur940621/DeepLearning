import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset

HIGH_RES = 96
LOW_RES = HIGH_RES // 4

data_transform = {
    "highres_transform":transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([HIGH_RES, HIGH_RES]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
   
    "lowres_transform":transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize([LOW_RES, LOW_RES], interpolation=Image.BICUBIC),
                              transforms.ToTensor(),
                              transforms.Normalize([0, 0, 0], [1, 1, 1])])
}

class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
 
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        high_res = data_transform["highres_transform"](image)
        low_res = data_transform["lowres_transform"](image)

        return low_res, high_res