import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

data_transform = {
    "input":transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([256, 256]),  
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
   
    "target":transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize([256, 256]),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
 

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :512, :]
        target_image = image[:, 512:, :]

        
        input_image = data_transform["input"](input_image)
        target_image = data_transform["target"](target_image)

        return input_image, target_image