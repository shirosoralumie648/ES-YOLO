#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# This script uses the prophesee-io library, which should be installed in the main environment.
# It is NOT part of the ReYOLOv8 submodule.
try:
    import aedat
except ImportError:
    print("错误: 找不到 'aedat' 包。", file=sys.stderr)
    print("请确保您已在活动的Python环境中安装了 'aedat' (pip install aedat)。", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
EVENT_DATA_PATH = r"c:\shirosoralumie648\ES-YOLO\Event_Data"

def convert_aedat4_to_npy(aedat4_path):
    """Reads an .aedat4 file and saves its contents as a .npy file using the aedat package."""
    try:
        decoder = aedat.Decoder(str(aedat4_path))
        
        # Collect all event packets from the decoder
        all_events = [packet['events'] for packet in decoder if 'events' in packet]

        if not all_events:
            tqdm.write(f"警告: {aedat4_path} 中没有找到事件包。")
            return False

        # Concatenate all event arrays into one
        events_raw = np.concatenate(all_events)

        # Create the new structured array with the correct dtype for PSEELoader
        # PSEELoader dtype: [('x', '<u2'), ('y', '<u2'), ('p', 'u1'), ('t', '<i8')]
        # aedat dtype:     [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
        structured_events = np.empty(events_raw.shape, dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'u1'), ('t', '<i8')])
        
        structured_events['x'] = events_raw['x']
        structured_events['y'] = events_raw['y']
        structured_events['p'] = events_raw['on'].astype('u1')  # Convert boolean 'on' to integer 'p'
        structured_events['t'] = events_raw['t'].astype('<i8') # Convert unsigned 't' to signed 't'

        npy_path = aedat4_path.with_suffix('.npy')
        np.save(npy_path, structured_events)
        return True
    except Exception as e:
        tqdm.write(f"转换失败 {aedat4_path}: {e}")
        return False

def main():
    """Main function to find and convert all .aedat4 files."""
    print("--- 开始将 .aedat4 文件转换为 .npy 格式 ---")
    aedat4_files = sorted(list(Path(EVENT_DATA_PATH).glob("**/*.aedat4")))

    if not aedat4_files:
        print("错误: 在指定的路径中没有找到 .aedat4 文件。", file=sys.stderr)
        print(f"检查路径: {EVENT_DATA_PATH}", file=sys.stderr)
        return

    print(f"找到了 {len(aedat4_files)} 个 .aedat4 文件需要转换。")

    success_count = 0
    for file_path in tqdm(aedat4_files, desc="转换进度"):
        if convert_aedat4_to_npy(file_path):
            success_count += 1
            os.remove(file_path) # Delete the original .aedat4 file to save space

    print(f"\n--- 转换完成! ---")
    print(f"成功转换 {success_count} / {len(aedat4_files)} 个文件。")

if __name__ == "__main__":
    main()
