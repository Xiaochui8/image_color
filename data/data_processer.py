import os
from PIL import Image
import shutil
import argparse
import tqdm
import random

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process(images, image_folder, target_folder):
    color_folder = os.path.join(target_folder, 'color')
    gray_folder = os.path.join(target_folder, 'gray')
    check_dir(color_folder)
    check_dir(gray_folder)
    
    for file_name in tqdm.tqdm(images):
        file_path = os.path.join(image_folder, file_name)
    
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(color_folder, file_name))

            # 打开图片并转换为灰度图
            img = Image.open(file_path)
            gray_img = img.convert('L')

            gray_img.save(os.path.join(gray_folder, file_name))   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='image')
    parser.add_argument('--train_folder', type=str, default='train')
    parser.add_argument('--valid_folder', type=str, default='valid')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--random_seed', type=int, default=6657)
    args = parser.parse_args()
    random.seed(args.random_seed)
    
    image_folder = args.image_folder
    train_folder = args.train_folder
    valid_folder = args.valid_folder
    # train_color_folder = os.path.join(train_folder, 'color')
    # train_gray_folder = os.path.join(train_folder, 'gray')
    # valid_color_folder = os.path.join(valid_folder, 'color')
    # valid_gray_folder = os.path.join(valid_folder, 'gray')
    
    # check_dir(train_color_folder)
    # check_dir(train_gray_folder)
    # check_dir(valid_color_folder)
    # check_dir(valid_gray_folder)
    
    images = os.listdir(image_folder)
    random.shuffle(images)
    train_num = int(len(images) * args.train_ratio)
    train_images = images[:train_num]
    valid_images = images[train_num:]
    
    process(train_images, image_folder, train_folder)
    process(valid_images, image_folder, valid_folder)
    
    
    
     