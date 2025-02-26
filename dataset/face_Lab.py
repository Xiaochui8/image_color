import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
from skimage.color import rgb2lab, rgb2gray

class ColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.color_dir = os.path.join(root_dir, 'color')
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.transform = transform
        self.images = os.listdir(self.color_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        color_image_name = os.path.join(self.color_dir, self.images[idx])
        color_image = Image.open(color_image_name)
        color_image = self.transform(color_image)
        gt_image = color_image.clone()
        color_image = color_image.permute(1, 2, 0)
        
        color_Lab = torch.tensor(rgb2lab(color_image)).permute(2, 0, 1)
        color_L = torch.tensor(rgb2gray(color_image)) # [0, 1]
        color_ab = (color_Lab[1:3, :, :] + 128) / 255    # [-128, 127] -> [0, 1]
        return color_L.unsqueeze(0), color_ab, gt_image # (1, h, w) (2, h, w)