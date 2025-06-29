#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from pathlib import Path
import sys

# --- Configuration ---
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DATASET_PATH = r"c:\shirosoralumie648\ES-YOLO\RGB_Dataset_RAW"
OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\ROI_Dataset_512"
ORIGINAL_SIZE = 640
ROI_SIZE = 512
MOVE_STEP = 1

CATEGORY_TO_ID = {
    "crazing": 0, "inclusion": 1, "patches": 2,
    "pitted_surface": 3, "rolled-in_scale": 4, "scratches": 5
}

# --- Core Functions ---

def setup_output_dirs():
    """Clears and creates the necessary output directories."""
    if Path(OUTPUT_PATH).exists():
        print(f"清空已存在的输出目录: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)
    
    for split in ["train", "validation"]:
        Path(OUTPUT_PATH, split, "images").mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_PATH, split, "labels").mkdir(parents=True, exist_ok=True)
    print("输出目录已准备就绪。")

def convert_xml_to_yolo_boxes(xml_path, img_width, img_height):
    """Parses an XML file and converts bounding boxes to YOLO format."""
    if not Path(xml_path).exists():
        return []
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CATEGORY_TO_ID:
                continue
            class_id = CATEGORY_TO_ID[name]
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            boxes.append([class_id, x_center, y_center, width, height])
    except ET.ParseError as e:
        print(f"\n警告: 解析XML文件失败 {xml_path}: {e}")
    return boxes

def crop_and_adjust_bbox(img, original_boxes, x_offset, y_offset):
    """Crops the image and adjusts the bounding boxes for the new ROI."""
    img_h, img_w = img.shape[:2]
    cropped_img = img[y_offset:y_offset + ROI_SIZE, x_offset:x_offset + ROI_SIZE]
    
    new_boxes = []
    for box in original_boxes:
        class_id, x_center, y_center, width, height = box
        
        abs_x_center = x_center * img_w
        abs_y_center = y_center * img_h
        abs_width = width * img_w
        abs_height = height * img_h
        abs_xmin = abs_x_center - abs_width / 2
        abs_ymin = abs_y_center - abs_height / 2
        
        new_xmin = abs_xmin - x_offset
        new_ymin = abs_ymin - y_offset
        new_xmax = new_xmin + abs_width
        new_ymax = new_ymin + abs_height
        
        intersect_xmin = max(0, new_xmin)
        intersect_ymin = max(0, new_ymin)
        intersect_xmax = min(ROI_SIZE, new_xmax)
        intersect_ymax = min(ROI_SIZE, new_ymax)
        
        if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
            new_box_x_center = (intersect_xmin + intersect_xmax) / (2 * ROI_SIZE)
            new_box_y_center = (intersect_ymin + intersect_ymax) / (2 * ROI_SIZE)
            new_box_width = (intersect_xmax - intersect_xmin) / ROI_SIZE
            new_box_height = (intersect_ymax - intersect_ymin) / ROI_SIZE
            
            if new_box_width * ROI_SIZE >= 2 and new_box_height * ROI_SIZE >= 2:
                new_boxes.append([class_id, new_box_x_center, new_box_y_center, new_box_width, new_box_height])
                
    return cropped_img, new_boxes

def process_split(split="train"):
    """Processes a dataset split (train or validation)."""
    print(f"\n--- 正在处理 {split} 数据集 ---")
    
    img_base_dir = Path(SRC_DATASET_PATH, split, "images")
    annotation_base_dir = Path(SRC_DATASET_PATH, split, "annotations")
    
    if not img_base_dir.exists():
        print(f"警告: 找不到目录 {img_base_dir}")
        return
        
    image_paths = []
    for category_dir in img_base_dir.iterdir():
        if category_dir.is_dir():
            image_paths.extend(list(category_dir.glob("*.jpg")))
            image_paths.extend(list(category_dir.glob("*.png")))

    total_processed = 0
    total_crops = 0

    for img_path in tqdm(image_paths, desc=f"处理 {split} 集"):
        base_name = img_path.stem
        xml_path = annotation_base_dir / f"{base_name}.xml"
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        if img.shape[:2] != (ORIGINAL_SIZE, ORIGINAL_SIZE):
            img = cv2.resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE))
            
        original_boxes = convert_xml_to_yolo_boxes(xml_path, ORIGINAL_SIZE, ORIGINAL_SIZE)
        if not original_boxes: continue
            
        max_offset = ORIGINAL_SIZE - ROI_SIZE
        crops_from_this_image = 0
        for offset in range(0, max_offset + 1, MOVE_STEP):
            x_offset = y_offset = offset
            
            cropped_img, new_boxes = crop_and_adjust_bbox(img, original_boxes, x_offset, y_offset)
            
            if not new_boxes: continue
            
            crop_name = f"{base_name}_diag_{offset}"
            output_img_path = Path(OUTPUT_PATH, split, "images", f"{crop_name}.jpg")
            output_label_path = Path(OUTPUT_PATH, split, "labels", f"{crop_name}.txt")
            
            cv2.imwrite(str(output_img_path), cropped_img)
            
            with open(output_label_path, 'w', encoding='utf-8') as f:
                for box in new_boxes:
                    class_id, x_c, y_c, w, h = box
                    f.write(f"{int(class_id)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            crops_from_this_image += 1
        
        if crops_from_this_image > 0:
            total_processed += 1
            total_crops += crops_from_this_image

    print(f"处理完成: 从 {total_processed} 张原始图像中生成了 {total_crops} 个有效裁剪。")

def main():
    """Main function to run the script."""
    print("开始数据集裁剪与标注转换...")
    setup_output_dirs()
    process_split("train")
    process_split("validation")
    print("\n所有处理完成!")

if __name__ == "__main__":
    main()

