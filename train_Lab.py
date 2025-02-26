import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import image_color.model_Lab as image_color
import argparse
import os
import dataset.face_Lab as face
from skimage.color import lab2rgb
from utils.Lab2rgb import Lab2rgb

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, labels):
        # 自定义损失计算
        loss = torch.mean((outputs - labels) ** 2)  # 示例：均方误差
        return loss

def train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

def valid(model, valid_loader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels, images in tqdm(valid_loader, desc="Validating"):
            inputs, labels, images = inputs.to(device), labels.to(device), images.to(device)

            outputs = model(inputs)
            # rgb_gt = Lab2rgb(inputs, labels)
            rgb_gt = images
            rgb_pred = torch.Tensor(Lab2rgb(inputs, outputs)).to(device)
            loss = criterion(rgb_gt, rgb_pred)

            running_loss += loss.item()

    print(f"Valid Loss: {running_loss/len(valid_loader)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='data/train')
    parser.add_argument('--valid_data_path', type=str, default='data/valid')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--model_path', type=str, default='model/image_color_Lab.pth')
    args = parser.parse_args()
    
    
    model = image_color.ImageColorNet().to(args.device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = face.ColorDataset(args.train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = face.ColorDataset(args.valid_data_path, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    

    train(model, train_loader, criterion, optimizer, num_epochs=args.num_epochs, device=args.device)
    valid(model, valid_loader, criterion, device=args.device)
    torch.save(model.state_dict(), args.model_path)
    