#!/usr/bin/env python

import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse

def load_image(image_path):
    """
    加载图像
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return image

def save_image(image, output_path):
    """
    保存图像
    """
    cv2.imwrite(str(output_path), image)

def get_center_square_mask(image_shape, square_size):
    """
    获取中心正方形掩码
    """
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

def remove_center_fill_background(source_image, square_size):
    """
    移除源图像中心正方形区域，用周围背景进行填充
    """
    h, w = source_image.shape[:2]
    center_h, center_w = h // 2, w // 2
    
    half_size = square_size // 2
    top = max(0, center_h - half_size)
    bottom = min(h, center_h + half_size)
    left = max(0, center_w - half_size)
    right = min(w, center_w + half_size)
    
    # 创建结果图像的副本
    result = source_image.copy()
    
    # 获取中心正方形掩码（反转掩码，保留背景区域）
    mask = get_center_square_mask(source_image.shape, square_size)
    mask_inv = cv2.bitwise_not(mask)
    
    # 提取背景区域
    background = cv2.bitwise_and(source_image, source_image, mask=mask_inv)
    
    # 用周围背景填充中心区域
    # 简单方法：用临近像素外推填充中心区域
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 通过模糊处理背景来模拟填充效果
    # 先应用掩码去除中心区域
    masked_img = np.where(mask_3ch == 255, 0, source_image)
    
    # 使用inpaint算法填充中心区域
    gray_mask = mask
    filled = cv2.inpaint(masked_img, gray_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return filled

def get_class_directories(base_path):
    """
    获取所有类别目录
    """
    return sorted([d for d in base_path.iterdir() if d.is_dir()])

def get_image_files(class_dir):
    """
    获取类别目录下的所有图像文件
    """
    return [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpeg', '.jpg', '.png']]

def main(square_size=50):
    # 定义路径
    attack_data_path = Path("datasets/MSTAR/ATTACK")
    output_path = Path("attack_result/RPL4")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别目录
    class_dirs = get_class_directories(attack_data_path)
    print(f"找到 {len(class_dirs)} 个类别目录")
    
    # 为每个类别创建处理后的样本
    for i, class_dir in enumerate(class_dirs):
        print(f"处理类别: {class_dir.name}")
        
        # 创建输出类别目录
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(exist_ok=True)
        
        # 获取当前类别的所有图像
        image_files = get_image_files(class_dir)
        
        # 处理当前类别的每张图像
        for image_file in image_files:
            try:
                # 加载源图像
                source_image = load_image(image_file)
                
                # 生成新的图像（移除中心区域，用背景填充）
                result_image = remove_center_fill_background(source_image, square_size)
                
                # 保存结果
                output_file = output_class_dir / image_file.name
                save_image(result_image, output_file)
                
            except Exception as e:
                print(f"  处理图像 {image_file.name} 时出错: {e}")

    print("所有图像处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成RPL对抗样本（移除中心区域，用背景填充）")
    parser.add_argument("--square_size", type=int, default=50, help="中心正方形区域的边长 (默认: 50)")
    args = parser.parse_args()
    
    main(args.square_size)