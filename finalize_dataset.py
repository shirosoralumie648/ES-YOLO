#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import hdf5plugin
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import datetime

# 禁用HDF5文件锁定，解决BlockingIOError: [Errno 11]问题
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# 添加REYOLOv8目录到路径以导入其formats_utils模块
sys.path.append(r"c:\shirosoralumie648\ES-YOLO\ReYOLOv8")
try:
    import formats_utils as fmt
except ImportError:
    print("错误: 找不到REYOLOv8的formats_utils模块，请确保路径正确")
    sys.exit(1)

# --- 事件数据配置 ---
EVENT_DATA_PATH = r"c:\shirosoralumie648\ES-YOLO\Event_Data"
LABELS_PATH = r"c:\shirosoralumie648\ES-YOLO\Processed_Videos_and_Labels"
FINAL_DATASET_DIR = r"c:\shirosoralumie648\ES-YOLO\ReYOLOv8_Dataset"

# --- 事件编码配置 ---
TIME_WINDOW_MS = 50  # 时间窗口，单位毫秒
TBIN = 10            # 通道数量
METHOD = "vtei"      # 编码方法：vtei, mdes, voxel_grid, shist
HEIGHT = 512         # 事件帧高度
WIDTH = 512          # 事件帧宽度

# --- 并行处理配置 ---
NUM_WORKERS = os.cpu_count()

def vtei_encode_events(events_npy, height=HEIGHT, width=WIDTH, tbin=TBIN):
    """
    将结构化的事件数据转换为VTEI格式的多通道帧表示。
    使用REYOLOv8的formats_utils模块中的vtei函数。
    
    参数:
        events_npy: 结构化的事件数组，包含x, y, p, t字段
        height: 输出帧的高度
        width: 输出帧的宽度
        tbin: 时间通道数
    
    返回:
        VTEI编码的多通道帧，形状为(tbin, height, width)
        时间戳数组，表示每个帧的时间戳
    """
    # 检查events_npy是否为空
    if events_npy.size == 0:
        print("警告: 事件数组为空，无法进行VTEI编码。")
        return np.zeros((tbin, height, width)), np.array([])
        
    # 获取事件的时间范围
    t_min, t_max = np.min(events_npy["t"]), np.max(events_npy["t"])
    total_time_us = t_max - t_min
    
    # 从events.npy中读取事件数据并转换为torch张量
    x = torch.from_numpy(np.clip(events_npy["x"].astype(np.int64).copy(), 0, width-1))
    y = torch.from_numpy(np.clip(events_npy["y"].astype(np.int64).copy(), 0, height-1))
    t = torch.from_numpy(events_npy["t"].astype(np.int64))
    p = torch.from_numpy(events_npy["p"].astype(np.int64).copy())
    
    print(f"事件数据统计: 事件总数={len(x)}, 时间范围={t_min}~{t_max} (总计{total_time_us/1e6:.2f}秒)")
    
    # 使用formats_utils中的vtei函数进行编码
    img = fmt.vtei(x, y, t, p, tbin, height, width).numpy()
    
    # 计算每一帧的大致时间戳（均匀划分时间区间）
    frame_timestamps = np.linspace(t_min, t_max, img.shape[0])
    
    return img, frame_timestamps

def process_event_file(video_dir, split):
    """
    处理一个视频的事件数据文件和对应的标签文件，
    将事件数据编码为VTEI格式，并返回数据供主进程写入。
    
    参数:
        video_dir: 包含events.npy的视频目录路径
        split: 数据集分割（train, validation, test）
        
    返回:
        video_name: 视频名称
        event_frames: 编码后的事件帧序列
        frame_timestamps: 对应每个事件帧的时间戳
        labels_data: 对应的标签数据
        stats: 处理统计信息
    """
    try:
        # 获取视频名称（目录名）
        video_name = os.path.basename(video_dir)
        tqdm.write(f"\n处理视频: {video_name}")
        
        # 构建事件数据和标签文件的完整路径
        event_file_path = os.path.join(video_dir, "events.npy")
        label_file_path = os.path.join(LABELS_PATH, split, "labels", f"{video_name}.npy")
        
        # 检查文件是否存在
        if not os.path.exists(event_file_path):
            tqdm.write(f"警告: 找不到事件文件 {event_file_path}，跳过处理。")
            return video_name, None, None, None, {"error": "事件文件不存在"}
        
        if not os.path.exists(label_file_path):
            tqdm.write(f"警告: 找不到标签文件 {label_file_path}，跳过处理。")
            return video_name, None, None, None, {"error": "标签文件不存在"}
            
        # 读取事件数据
        events_npy = np.load(event_file_path)
        tqdm.write(f"已读取事件文件: {event_file_path}, 事件数量: {len(events_npy)}")
        
        # VTEI编码事件数据
        event_frames, frame_timestamps = vtei_encode_events(events_npy)
        
        # 读取标签数据
        labels_data = np.load(label_file_path)
        tqdm.write(f"已读取标签文件: {label_file_path}, 标签形状: {labels_data.shape}")
        
        # 收集统计信息
        stats = {
            "video_name": video_name,
            "event_count": len(events_npy),
            "frame_count": len(event_frames),
            "label_count": len(labels_data),
            "frame_timestamps": frame_timestamps,
        }
        
        # 验证帧和标签对应关系
        if len(labels_data) > 0:
            # 检查标签内容的有效性
            has_valid_labels = np.all(np.isfinite(labels_data)) and not np.any(np.isnan(labels_data))
            if not has_valid_labels:
                tqdm.write(f"警告: {video_name} 的标签数据包含NaN或Inf值")
        
        return video_name, event_frames, frame_timestamps, labels_data, stats
    except Exception as e:
        tqdm.write(f"处理 {video_dir} 时出错: {e}")
        import traceback
        tqdm.write(traceback.format_exc())
        return video_name, None, None, None, {"error": str(e)}

def main():
    """
    主函数：处理所有视频的事件数据和标签，创建最终的h5格式数据集。
    按照REYOLOv8的要求，使用VTEI编码事件数据，并确保h5文件结构正确。
    每个视频都会成为h5文件中的一个组(group)，每个组中包含多帧事件数据和对应的标签。
    """
    print(f"\n--- 开始处理数据集 ---")
    print(f"事件编码方法: {METHOD}")
    print(f"帧尺寸: {HEIGHT}x{WIDTH}")
    print(f"通道数: {TBIN}")
    
    # 1. 设置输出目录
    Path(FINAL_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件
    log_path = os.path.join(FINAL_DATASET_DIR, "dataset_creation_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"时间: {datetime.datetime.now()}\n")
        log_file.write(f"编码方法: {METHOD}\n")
        log_file.write(f"帧尺寸: {HEIGHT}x{WIDTH}\n")
        log_file.write(f"通道数: {TBIN}\n\n")

    for split in ['train', 'validation']:
        print(f"\n--- 处理 {split} 数据集 ---")
        
        # 查找此分割中的所有视频目录
        split_dir = os.path.join(EVENT_DATA_PATH, split)
        if not os.path.exists(split_dir):
            print(f"警告: 找不到分割目录 {split_dir}，跳过处理。")
            continue

        # 查找包含events.npy的所有视频目录
        video_dirs = []
        for d in os.listdir(split_dir):
            full_dir = os.path.join(split_dir, d)
            if os.path.isdir(full_dir) and os.path.exists(os.path.join(full_dir, "events.npy")):
                video_dirs.append(full_dir)
        
        if not video_dirs:
            print(f"在 {split_dir} 中没有找到包含 events.npy 的视频目录，跳过处理。")
            continue

        print(f"在 {split_dir} 中找到 {len(video_dirs)} 个视频目录")
        
        output_path = os.path.join(FINAL_DATASET_DIR, f"{split}.h5")
        log_split_path = os.path.join(FINAL_DATASET_DIR, f"{split}_stats.txt")
        
        # 创建此分割的空H5文件
        with h5py.File(output_path, 'w') as hf, open(log_split_path, 'w') as log_file:
            log_file.write(f"--- {split} 数据集统计 ---\n\n")
            
            # 记录视频处理统计信息
            video_stats = {}
            processed_count = 0
            skipped_count = 0
            total_frames = 0
            problem_videos = []
            
            # 使用多进程并行读取事件文件和对应的标签文件
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # 向进程池提交所有文件读取任务
                future_to_dir = {executor.submit(process_event_file, video_dir, split): video_dir 
                               for video_dir in video_dirs}
                
                # 处理结果
                for future in tqdm(as_completed(future_to_dir), total=len(video_dirs), desc=f"处理 {split} 数据集"):
                    try:
                        # 获取处理后的视频名称和数据
                        video_name, event_frames, frame_timestamps, labels_data, stats = future.result()
                        
                        # 记录此视频的统计信息
                        video_stats[video_name] = stats
                        
                        # 如果处理失败，记录错误并跳过
                        if "error" in stats:
                            skipped_count += 1
                            problem_videos.append(f"{video_name}: {stats['error']}")
                            log_file.write(f"视频 {video_name} 处理失败: {stats['error']}\n")
                            continue
                        
                        # 确保工作进程返回的数据有效
                        if video_name and event_frames is not None and labels_data is not None:
                            # 为视频创建一个组
                            video_group = hf.create_group(video_name)
                            
                            # 记录帧数和标签数
                            frame_count = len(event_frames)
                            label_count = len(labels_data)
                            total_frames += frame_count
                            
                            # 记录时间戳范围
                            if len(frame_timestamps) > 0:
                                min_ts = frame_timestamps[0]
                                max_ts = frame_timestamps[-1]
                                # 将时间戳范围保存到组属性
                                video_group.attrs["min_timestamp"] = min_ts
                                video_group.attrs["max_timestamp"] = max_ts
                                log_file.write(f"视频 {video_name}: 帧数={frame_count}, 标签数={label_count}, 时间戳范围={min_ts}~{max_ts}\n")
                            else:
                                log_file.write(f"视频 {video_name}: 帧数={frame_count}, 标签数={label_count}, 无时间戳\n")
                            
                            # 写入每一帧，确保帧号正确
                            for frame_idx, frame_data in enumerate(event_frames):
                                # 使用实际帧号索引作为数据集名称
                                frame_name = str(frame_idx)
                                
                                # 创建数据集
                                try:
                                    video_group.create_dataset(
                                        frame_name,
                                        data=frame_data,
                                        **hdf5plugin.Blosc()
                                    )
                                except Exception as e:
                                    log_file.write(f"  警告: 无法写入帧 {frame_name}: {e}\n")
                            
                            # 将整个视频的标签写入同一个组
                            try:
                                video_group.create_dataset('labels', data=labels_data, **hdf5plugin.Blosc())
                            except Exception as e:
                                log_file.write(f"  警告: 无法写入标签: {e}\n")
                                
                            processed_count += 1
                            
                    except Exception as e:
                        # 记录处理过程中发生的任何异常
                        import traceback
                        error_trace = traceback.format_exc()
                        tqdm.write(f"池中的任务引发异常: {e}\n{error_trace}")
                        log_file.write(f"处理 {video_name} 时出错: {e}\n{error_trace}\n")

            # 记录分割的总体统计信息
            log_file.write(f"\n--- 汇总统计 ---\n")
            log_file.write(f"总共处理视频: {len(video_dirs)} 个\n")
            log_file.write(f"成功处理视频: {processed_count} 个\n")
            log_file.write(f"跳过处理视频: {skipped_count} 个\n")
            log_file.write(f"总帧数: {total_frames} 帧\n")
            
            if problem_videos:
                log_file.write(f"\n问题视频列表:\n")
                for problem in problem_videos:
                    log_file.write(f"  - {problem}\n")
            
            # 打印当前分割的统计信息
            print(f"\n{split} 数据集统计:")
            print(f"  - 总共处理视频: {len(video_dirs)} 个")
            print(f"  - 成功处理视频: {processed_count} 个")
            print(f"  - 跳过处理视频: {skipped_count} 个")
            print(f"  - 总帧数: {total_frames} 帧")
            
    print("\n数据集最终处理完成！")
    print(f"最终数据集位于: {FINAL_DATASET_DIR}")
    print(f"详细日志文件: {os.path.join(FINAL_DATASET_DIR, 'dataset_creation_log.txt')}")
    print(f"分割统计文件:")
    for split in ['train', 'validation']:
        log_split_path = os.path.join(FINAL_DATASET_DIR, f"{split}_stats.txt")
        if os.path.exists(log_split_path):
            print(f"  - {log_split_path}")
            
    print("\n提示: 检查日志文件中是否有帧号连续性问题和标签格式问题，以解决训练时的'丢帧'错误。")
    print("如果发现帧号不连续，请考虑修改 EventVideoDataset.__getitem__ 方法，增加帧缺失保护机制。")

if __name__ == "__main__":
    main()
