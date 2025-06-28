#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from pathlib import Path

# 配置参数
SRC_DATASET_PATH = r"c:\shirosoralumie648\ES-YOLO\RGB_Dataset_RAW"
OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\ROI_Dataset_512"
ORIGINAL_SIZE = 640  # 原图尺寸
ROI_SIZE = 512      # 裁剪的ROI尺寸
MOVE_STEP = 1       # 每次移动的像素

# 创建输出目录
def create_output_dirs():
    # 训练集
    Path(f"{OUTPUT_PATH}/train/images").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_PATH}/train/labels").mkdir(parents=True, exist_ok=True)
    
    # 验证集
    Path(f"{OUTPUT_PATH}/validation/images").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_PATH}/validation/labels").mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录创建完成: {OUTPUT_PATH}")

# 将XML格式的边界框转换为YOLO格式
def convert_bbox_xml_to_yolo(xml_file, img_width, img_height):
    """
    将XML格式的边界框转换为YOLO格式
    YOLO格式: [class_id, x_center, y_center, width, height] 
    其中所有值都归一化为0-1之间
    """
    boxes = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            # 获取类别
            category = obj.find('n').text
            
            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 根据类别获取类别ID
            if category == "crazing":
                class_id = 0
            elif category == "inclusion":
                class_id = 1
            elif category == "patches":
                class_id = 2
            elif category == "pitted_surface":
                class_id = 3
            elif category == "rolled-in_scale":
                class_id = 4
            elif category == "scratches":
                class_id = 5
            else:
                print(f"未知类别: {category}")
                continue
                
            # 转换为YOLO格式 (中心点x, 中心点y, 宽度, 高度) - 归一化为0-1
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 确保所有值在0-1范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            boxes.append([class_id, x_center, y_center, width, height])
    except Exception as e:
        print(f"解析XML文件出错: {e}")
    
    return boxes

# 裁剪图像并调整YOLO格式的边界框
def crop_and_adjust_bbox(img, boxes, x_offset, y_offset, roi_size):
    """
    裁剪图像并调整YOLO格式的边界框
    
    参数:
    - img: 原始图像
    - boxes: YOLO格式的边界框 [class_id, x_center, y_center, width, height]
    - x_offset, y_offset: 裁剪的偏移量（以像素为单位）
    - roi_size: ROI的大小
    
    返回:
    - cropped_img: 裁剪后的图像
    - new_boxes: 调整后的边界框
    """
    img_height, img_width = img.shape[:2]
    
    # 裁剪图像
    cropped_img = img[y_offset:y_offset+roi_size, x_offset:x_offset+roi_size]
    
    # 调整边界框
    new_boxes = []
    
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        # 将归一化坐标转换为像素坐标
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height
        
        # 计算边界框在原图中的绝对坐标
        abs_xmin = abs_x_center - abs_width / 2
        abs_ymin = abs_y_center - abs_height / 2
        abs_xmax = abs_x_center + abs_width / 2
        abs_ymax = abs_y_center + abs_height / 2
        
        # 计算边界框在裁剪区域中的绝对坐标
        crop_xmin = abs_xmin - x_offset
        crop_ymin = abs_ymin - y_offset
        crop_xmax = abs_xmax - x_offset
        crop_ymax = abs_ymax - y_offset
        
        # 如果边界框完全在裁剪区域外，则跳过
        if crop_xmax <= 0 or crop_ymax <= 0 or crop_xmin >= roi_size or crop_ymin >= roi_size:
            continue
        
        # 如果边界框部分在裁剪区域内，则调整边界
        crop_xmin = max(0, crop_xmin)
        crop_ymin = max(0, crop_ymin)
        crop_xmax = min(roi_size, crop_xmax)
        crop_ymax = min(roi_size, crop_ymax)
        
        # 计算新的中心点和尺寸（归一化到0-1）
        new_x_center = (crop_xmin + crop_xmax) / (2 * roi_size)
        new_y_center = (crop_ymin + crop_ymax) / (2 * roi_size)
        new_width = (crop_xmax - crop_xmin) / roi_size
        new_height = (crop_ymax - crop_ymin) / roi_size
        
        # 确保所有值在0-1范围内
        new_x_center = max(0, min(1, new_x_center))
        new_y_center = max(0, min(1, new_y_center))
        new_width = max(0, min(1, new_width))
        new_height = max(0, min(1, new_height))
        
        # 排除过小的边界框（可选）
        if new_width * roi_size < 2 or new_height * roi_size < 2:
            continue
            
        new_boxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return cropped_img, new_boxes

# 处理数据集
def process_dataset(dataset_type="train"):
    print(f"正在处理 {dataset_type} 数据集...")
    
    # 获取所有类别文件夹
    img_base_dir = os.path.join(SRC_DATASET_PATH, dataset_type, "images")
    annotation_base_dir = os.path.join(SRC_DATASET_PATH, dataset_type, "annotations")
    
    # 获取所有类别
    categories = [d for d in os.listdir(img_base_dir) if os.path.isdir(os.path.join(img_base_dir, d))]
    
    total_processed = 0
    images_with_no_boxes = 0
    
    for category in categories:
        # 获取该类别下的所有图片
        img_dir = os.path.join(img_base_dir, category)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        for img_file in tqdm(img_files, desc=f"处理 {category}"):
            # 图片和标注文件路径
            img_path = os.path.join(img_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            xml_path = os.path.join(annotation_base_dir, f"{base_name}.xml")
            
            # 检查XML文件是否存在
            if not os.path.exists(xml_path):
                print(f"警告: 找不到标注文件 {xml_path}")
                continue
            
            # 读取图像并调整大小
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
                
            # 如果原始图片大小不是640x640，调整为640x640
            if img.shape[0] != ORIGINAL_SIZE or img.shape[1] != ORIGINAL_SIZE:
                img = cv2.resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE))
                
            # 从XML获取边界框
            boxes = convert_bbox_xml_to_yolo(xml_path, ORIGINAL_SIZE, ORIGINAL_SIZE)
            if len(boxes) == 0:
                print(f"警告: 图像 {img_path} 没有有效的边界框")
                continue
                
            # 裁剪图像和调整边界框
            crop_count = 0
            for y_offset in range(0, ORIGINAL_SIZE - ROI_SIZE + 1, MOVE_STEP):
                for x_offset in range(0, ORIGINAL_SIZE - ROI_SIZE + 1, MOVE_STEP):
                    # 裁剪图像并调整边界框
                    cropped_img, new_boxes = crop_and_adjust_bbox(img, boxes, x_offset, y_offset, ROI_SIZE)
                    
                    # 如果没有边界框，跳过保存
                    if len(new_boxes) == 0:
                        images_with_no_boxes += 1
                        continue
                    
                    # 保存裁剪后的图像
                    output_img_path = os.path.join(OUTPUT_PATH, dataset_type, "images", f"{base_name}_x{x_offset}_y{y_offset}.jpg")
                    cv2.imwrite(output_img_path, cropped_img)
                    
                    # 保存调整后的YOLO格式标注
                    output_label_path = os.path.join(OUTPUT_PATH, dataset_type, "labels", f"{base_name}_x{x_offset}_y{y_offset}.txt")
                    with open(output_label_path, 'w') as f:
                        for box in new_boxes:
                            class_id, x_center, y_center, width, height = box
                            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    crop_count += 1
                    total_processed += 1
            
            print(f"图像 {img_file} 已处理，生成了 {crop_count} 个裁剪")
    
    print(f"处理完毕! 总共处理了 {total_processed} 张图像")
    print(f"有 {images_with_no_boxes} 张裁剪后的图像没有边界框")

def main():
    print("开始数据集裁剪与标注转换处理...")
    create_output_dirs()
    
    # 处理训练集
    process_dataset("train")
    
    # 处理验证集
    process_dataset("validation")
    
    print("所有处理完成!")

if __name__ == "__main__":
    main()
