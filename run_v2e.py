#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys
import concurrent.futures
import itertools

# --- Configuration ---
VIDEO_INPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\Processed_Videos_and_Labels"
EVENT_OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\Event_Data"
GPUS = [0, 1]  # 可用的GPU ID列表
WORKERS_PER_GPU = 6
MAX_WORKERS = len(GPUS) * WORKERS_PER_GPU

# v2e parameters
V2E_PARAMS = {
    "output_width": 512,
    "output_height": 512,
    "pos_thres": 0.15,
    "neg_thres": 0.15,
    "sigma_thres": 0.01,
    "cutoff_hz": 300,
    "leak_rate_hz": 0.1,
    "shot_noise_rate_hz": 0.001,
}

# --- Core Functions ---

def setup_output_dirs():
    """Creates the necessary output directories for events."""
    Path(EVENT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    print(f"事件数据输出目录已准备就绪: {EVENT_OUTPUT_PATH}")

def build_v2e_command(input_video, output_dir):
    """Builds the v2e command string from parameters."""
    command = [
        "v2e",
        "--ignore-gooey",  # Prevent GUI-related hangs
        "-i", str(input_video),
        "-o", str(output_dir),
        "--overwrite",
        "--dvs_aedat4", "events.aedat4",  # Correctly provide a filename
    ]

    for key, value in V2E_PARAMS.items():
        if value is not None:
            command.append(f"--{key}")
            command.append(str(value))

    return command

def run_v2e_task(video_path, output_base_dir, gpu_id):
    """Worker function to process a single video on a specific GPU."""
    tqdm.write(f"[GPU {gpu_id}] 开始处理: {video_path.name}")
    try:
        # Set environment variable for this specific process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        event_output_dir = output_base_dir / video_path.stem
        event_output_dir.mkdir(exist_ok=True)
        command = build_v2e_command(video_path, event_output_dir)
        
        # Run v2e in the modified environment
        subprocess.run(command, check=True, capture_output=True, env=env)
        tqdm.write(f"[GPU {gpu_id}] 完成: {video_path.name}")
        return None, video_path.name  # Success
    except subprocess.CalledProcessError as e:
        stderr_str = e.stderr.decode('utf-8', errors='replace')
        error_message = (
            f"\n--- ERROR on GPU {gpu_id} processing {video_path.name} ---\n"
            f"返回码: {e.returncode}\n"
            f"错误: {stderr_str}\n"
            f"--- END ERROR ---"
        )
        tqdm.write(error_message)
        return video_path.name, video_path.name  # Return filename on error
    except Exception as e:
        tqdm.write(f"处理 {video_path.name} on GPU {gpu_id} 时发生意外错误: {e}")
        return video_path.name, video_path.name

def process_split(split="train"):
    """Runs v2e on all videos for a given dataset split using a process pool and multi-GPU."""
    print(f"\n--- 正在为 {split} 数据集生成事件数据 (总并行数: {MAX_WORKERS}, 每张GPU: {WORKERS_PER_GPU}) ---")
    
    video_dir = Path(VIDEO_INPUT_PATH, split, "videos")
    output_base_dir = Path(EVENT_OUTPUT_PATH, split)
    
    if not video_dir.exists():
        print(f"警告: 找不到视频目录 {video_dir}")
        return

    video_paths = sorted(list(video_dir.glob("*.mp4")))
    if not video_paths:
        print(f"在 {video_dir} 中没有找到 .mp4 文件。")
        return

    output_base_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    gpu_cycle = itertools.cycle(GPUS)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Assign a GPU to each task in a round-robin fashion
        future_to_video = {
            executor.submit(run_v2e_task, path, output_base_dir, next(gpu_cycle)): path 
            for path in video_paths
        }
        
        # Process futures as they complete and show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_paths), desc=f"转换 {split} 视频"):
            error, video_name = future.result()
            if error:
                errors.append(video_name)

    if errors:
        print(f"\n处理完成，但有 {len(errors)} 个视频转换失败:")
        for error_file in errors:
            print(f"- {error_file}")
    else:
        print(f"\n{split} 数据集的所有视频已成功转换为事件数据。")

def main():
    """Main function to run the script."""
    print("开始使用 v2e 将视频转换为事件数据...")
    setup_output_dirs()
    try:
        print("正在检查 v2e 是否已安装...")
        result = subprocess.run(["v2e", "--ignore-gooey", "-h"], check=True, capture_output=True, text=True)
        print("v2e 已找到。")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n错误: v2e 未安装或无法在当前环境中运行。")
        print("请先通过 'pip install v2e' 安装它。")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"错误详情: {e.stderr}")
        return

    process_split("train")
    process_split("validation")
    print("\n所有事件数据生成完成!")

if __name__ == "__main__":
    main()
