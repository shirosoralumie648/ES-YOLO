#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from tqdm import tqdm
from pathlib import Path
import re
from collections import defaultdict

# --- Configuration ---
ROI_DATASET_PATH = r"c:\shirosoralumie648\ES-YOLO\ROI_Dataset_512"
VIDEO_OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\videos"
FRAME_RATE = 64
IMG_SIZE = (512, 512)

# --- Core Functions ---

def setup_video_output_dirs():
    """Creates the necessary output directories for videos."""
    for split in ["train", "validation"]:
        Path(VIDEO_OUTPUT_PATH, split).mkdir(parents=True, exist_ok=True)
    print(f"视频输出目录已准备就绪: {VIDEO_OUTPUT_PATH}")

def group_images_by_base_name(image_dir):
    """Groups image paths by their original base name."""
    image_groups = defaultdict(list)
    for img_path in image_dir.glob("*.jpg"):
        match = re.match(r"(.+)_diag_\d+", img_path.stem)
        if match:
            base_name = match.group(1)
            image_groups[base_name].append(img_path)
    return image_groups

def sort_images_numerically(image_paths):
    """Sorts image paths based on the numeric suffix (_diag_XX)."""
    return sorted(image_paths, key=lambda p: int(re.search(r"_diag_(\d+)", p.stem).group(1)))

def create_videos_for_split(split="train"):
    """Creates videos from image sequences for a given dataset split."""
    print(f"\n--- 正在为 {split} 数据集创建视频 ---")
    
    image_dir = Path(ROI_DATASET_PATH, split, "images")
    output_dir = Path(VIDEO_OUTPUT_PATH, split)
    
    if not image_dir.exists():
        print(f"警告: 找不到图像目录 {image_dir}")
        return

    image_groups = group_images_by_base_name(image_dir)
    
    if not image_groups:
        print(f"在 {image_dir} 中没有找到符合命名格式的图片。")
        return

    for base_name, img_paths in tqdm(image_groups.items(), desc=f"创建 {split} 视频"):
        sorted_paths = sort_images_numerically(img_paths)
        
        # Define the codec and create VideoWriter object
        # Using 'mp4v' for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_path = output_dir / f"{base_name}.mp4"
        
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, FRAME_RATE, IMG_SIZE)
        
        for img_path in sorted_paths:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                # Ensure frame is the correct size, just in case
                if frame.shape[:2] != IMG_SIZE[::-1]: # (height, width) vs (width, height)
                    frame = cv2.resize(frame, IMG_SIZE)
                video_writer.write(frame)
        
        video_writer.release()

    print(f"{split} 数据集的视频创建完成。")

def main():
    """Main function to run the script."""
    print("开始从图像序列创建视频...")
    setup_video_output_dirs()
    create_videos_for_split("train")
    create_videos_for_split("validation")
    print("\n所有视频处理完成!")

if __name__ == "__main__":
    main()
