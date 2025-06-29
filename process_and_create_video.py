#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from pathlib import Path
import sys

# --- Configuration ---
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DATASET_PATH = r"c:\shirosoralumie648\ES-YOLO\RGB_Dataset_RAW"
OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\Processed_Videos_and_Labels"
ORIGINAL_SIZE = 640
ROI_SIZE = 512
MOVE_STEP = 1
FRAME_RATE = 64

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
        Path(OUTPUT_PATH, split, "videos").mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_PATH, split, "labels").mkdir(parents=True, exist_ok=True)
    print("输出目录已准备就绪。")

def parse_xml_annotations(xml_path):
    """Parses an XML file to get object annotations."""
    if not Path(xml_path).exists():
        return []
    annotations = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for i, obj in enumerate(root.findall('object')):
            name = obj.find('name').text
            if name not in CATEGORY_TO_ID:
                continue
            class_id = CATEGORY_TO_ID[name]
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            annotations.append({
                'class_id': class_id, 'track_id': i,
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
            })
    except ET.ParseError as e:
        print(f"\n警告: 解析XML文件失败 {xml_path}: {e}")
    return annotations

def process_single_image(img_path, annotation_base_dir, video_out_dir, label_out_dir):
    """Processes a single image: creates a video and a corresponding .npy label file."""
    base_name = img_path.stem
    xml_path = annotation_base_dir / f"{base_name}.xml"
    
    img = cv2.imread(str(img_path))
    if img is None: return
    
    if img.shape[:2] != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        img = cv2.resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE))
        
    original_annotations = parse_xml_annotations(xml_path)
    if not original_annotations: return

    video_path = video_out_dir / f"{base_name}.mp4"
    label_path = label_out_dir / f"{base_name}.npy"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, FRAME_RATE, (ROI_SIZE, ROI_SIZE))
    
    bbox_events = []
    bbox_dtype = np.dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('track_id', '<u4')])

    max_offset = ORIGINAL_SIZE - ROI_SIZE
    for offset in range(0, max_offset + 1, MOVE_STEP):
        frame_idx = offset
        x_offset = y_offset = offset
        timestamp = frame_idx * (1_000_000 // FRAME_RATE)

        # Create and write frame
        frame = img[y_offset:y_offset + ROI_SIZE, x_offset:x_offset + ROI_SIZE]
        video_writer.write(frame)

        # Adjust boxes for the current frame
        for ann in original_annotations:
            new_xmin = ann['xmin'] - x_offset
            new_ymin = ann['ymin'] - y_offset
            new_xmax = ann['xmax'] - x_offset
            new_ymax = ann['ymax'] - y_offset

            intersect_xmin = max(0, new_xmin)
            intersect_ymin = max(0, new_ymin)
            intersect_xmax = min(ROI_SIZE, new_xmax)
            intersect_ymax = min(ROI_SIZE, new_ymax)

            if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
                w = intersect_xmax - intersect_xmin
                h = intersect_ymax - intersect_ymin
                if w >= 2 and h >= 2:
                    x_center = (intersect_xmin + intersect_xmax) / (2 * ROI_SIZE)
                    y_center = (intersect_ymin + intersect_ymax) / (2 * ROI_SIZE)
                    norm_w = w / ROI_SIZE
                    norm_h = h / ROI_SIZE
                    bbox_events.append((timestamp, x_center, y_center, norm_w, norm_h, ann['class_id'], ann['track_id']))

    video_writer.release()
    
    if bbox_events:
        structured_events = np.array(bbox_events, dtype=bbox_dtype)
        np.save(label_path, structured_events)

def process_split(split="train"):
    """Processes a dataset split (train or validation)."""
    print(f"\n--- 正在处理 {split} 数据集 ---")
    
    img_base_dir = Path(SRC_DATASET_PATH, split, "images")
    annotation_base_dir = Path(SRC_DATASET_PATH, split, "annotations")
    video_out_dir = Path(OUTPUT_PATH, split, "videos")
    label_out_dir = Path(OUTPUT_PATH, split, "labels")
    
    if not img_base_dir.exists():
        print(f"警告: 找不到目录 {img_base_dir}")
        return
        
    image_paths = []
    for category_dir in img_base_dir.iterdir():
        if category_dir.is_dir():
            image_paths.extend(list(category_dir.glob("*.jpg")))
            image_paths.extend(list(category_dir.glob("*.png")))

    for img_path in tqdm(image_paths, desc=f"处理 {split} 集"):
        process_single_image(img_path, annotation_base_dir, video_out_dir, label_out_dir)

    print(f"{split} 数据集处理完成。")

def main():
    """Main function to run the script."""
    print("开始一体化处理：直接从原图生成视频和标注...")
    setup_output_dirs()
    process_split("train")
    process_split("validation")
    print("\n所有处理完成!")

if __name__ == "__main__":
    main()
