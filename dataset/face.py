import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

class ColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.color_dir = os.path.join(root_dir, 'color')
        self.gray_dir = os.path.join(root_dir, 'gray')
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
        gray_image_name = os.path.join(self.gray_dir, self.images[idx])
        
        color_image = Image.open(color_image_name)
        gray_image = Image.open(gray_image_name).convert('L')
        color_image = self.transform(color_image)
        gray_image = self.transform(gray_image)
        return gray_image, color_image