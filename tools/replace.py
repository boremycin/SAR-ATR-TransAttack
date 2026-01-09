import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return image

def save_image(image, output_path):
    cv2.imwrite(str(output_path), image)

def get_center_square_mask(image_shape, square_size):
    h, w = image_shape[:2]
    center_h, center_w = h // 2, w // 2
    
    half_size = square_size // 2
    top = max(0, center_h - half_size)
    bottom = min(h, center_h + half_size)
    left = max(0, center_w - half_size)
    right = min(w, center_w + half_size)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask

def replace_background(source_image, background_image, square_size):
    if source_image.shape != background_image.shape:
        background_image = cv2.resize(background_image, (source_image.shape[1], source_image.shape[0]))
    mask = get_center_square_mask(source_image.shape, square_size)
    
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    result = np.where(mask_3ch == 255, source_image, background_image)
    return result

def get_class_directories(base_path):
    return sorted([d for d in base_path.iterdir() if d.is_dir()])

def get_image_files(class_dir):
    return [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpeg', '.jpg', '.png']]

def main(square_size=64):
    attack_data_path = Path("datasets/MSTAR/ATTACK")
    output_path = Path("attack_result/RPL2")
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_dirs = get_class_directories(attack_data_path)
    print(f"找到 {len(class_dirs)} 个类别目录")
    
    for i, class_dir in enumerate(class_dirs):
        print(f"处理类别: {class_dir.name}")
        
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(exist_ok=True)
        
        image_files = get_image_files(class_dir)
        
        next_class_index = (i + 1) % len(class_dirs)
        background_class_dir = class_dirs[next_class_index]
        background_images = get_image_files(background_class_dir)
        
        print(f"  使用类别 '{background_class_dir.name}' 的图像作为背景")
        
        for image_file in tqdm(image_files):
            try:
                source_image = load_image(image_file)

                background_image_file = random.choice(background_images)
                background_image = load_image(background_image_file)
        
                result_image = replace_background(source_image, background_image, square_size)
                
                output_file = output_class_dir / image_file.name
                save_image(result_image, output_file)
                
            except Exception as e:
                print(f"  处理图像 {image_file.name} 时出错: {e}")

    print("所有图像处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成RPL对抗样本")
    parser.add_argument("--square_size", type=int, default=50, help="中心正方形区域的边长 (默认: 64)")
    args = parser.parse_args()
    main(args.square_size)

