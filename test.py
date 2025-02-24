import torch
import image_color
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # 灰度图
    gray_image_path = 'data/test.jpg'
    gray_image = Image.open(gray_image_path).convert('L')
    transform = transforms.ToTensor()
    gray_image = transform(gray_image).unsqueeze(0).to(device) # (b, 1, h, w)
    
    model = image_color.ImageColorNet()
    # model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)
    
    image = model(gray_image)
    
    to_pil = transforms.ToPILImage()
    image = to_pil(image[0].cpu().detach())
    plt.imshow(image)
    plt.show()
    
    pass
    
    