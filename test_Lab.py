import torch
import image_color.model_Lab as image_color
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.Lab2rgb import Lab2rgb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # 灰度图
    gray_image_path = 'data/test.jpg'
    gray_image = Image.open(gray_image_path).convert('L')
    transform = transforms.ToTensor()
    gray_image = transform(gray_image).unsqueeze(0).to(device) # (b, 1, h, w)
    
    model = image_color.ImageColorNet()
    model.load_state_dict(torch.load('model/image_color_lab.pth'))
    model = model.eval().to(device)
    
    with torch.no_grad():
        image = model(gray_image)
        if image.shape[1] == 2:
            image = Lab2rgb(gray_image, image)
        
        to_pil = transforms.ToPILImage()
        image = to_pil(image[0].cpu().detach())
        image.save('data/test_result.jpg')
        plt.imshow(image)
        plt.show()
    
    pass
    
    