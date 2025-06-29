#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Ensure the script can find the ReYOLOv8 modules
sys.path.append(str(Path(__file__).parent / 'ReYOLOv8'))

from prophesee.io.psee_loader import PSEELoader
import formats_utils as fmt
import utils
import hdf5plugin

# --- Configuration ---
EVENT_DATA_PATH = r"c:\shirosoralumie648\ES-YOLO\Event_Data"
LABEL_PATH = r"c:\shirosoralumie648\ES-YOLO\Processed_Videos_and_Labels"
FINAL_OUTPUT_PATH = r"c:\shirosoralumie648\ES-YOLO\ReYOLOv8_Dataset"

# Processing Parameters (from singleShot_eventDataHandler_GEN1.py)
TIME_WINDOW_MS = 50  # Time window for each frame in milliseconds
IMG_WIDTH = 512
IMG_HEIGHT = 512
METHOD = "vtei"       # Event-to-frame conversion method
TBIN = 10             # Number of time bins for the method
FILTER_MIN_DIAG = 30  # Minimum diagonal size of a bounding box to keep
FILTER_MIN_SIDE = 10  # Minimum side size of a bounding box to keep

# --- Core Functions ---

def process_file_pair(event_file, label_file, temp_dest_folder, split, file_index):
    """Processes a single pair of event and label files to create sub-sequence H5 files."""
    video_loader = PSEELoader(str(event_file))
    boxes = np.load(label_file)

    delta_t = 1000 * TIME_WINDOW_MS
    sub_sequence_idx = 0

    while not video_loader.done:
        events = video_loader.load_delta_t(delta_t)
        if events.size == 0:
            continue

        start_time, end_time = events['t'][0], events['t'][-1]
        current_boxes = boxes[(boxes['t'] >= start_time) & (boxes['t'] <= end_time)]

        if current_boxes.size > 0:
            # Convert event chunk to a frame-like representation
            x = torch.from_numpy(np.clip(events["x"].astype(np.int64), 0, IMG_WIDTH - 1))
            y = torch.from_numpy(np.clip(events["y"].astype(np.int64), 0, IMG_HEIGHT - 1))
            t = torch.from_numpy(events["t"].astype(np.int64))
            p = torch.from_numpy(events["p"].astype(np.int64))

            if METHOD == "vtei":
                img_tensor = fmt.vtei(x, y, t, p, TBIN, IMG_HEIGHT, IMG_WIDTH)
            else:
                # Placeholder for other methods if needed
                raise NotImplementedError(f"Method {METHOD} is not implemented.")

            # The old script expected pixel coords, but our .npy are already normalized.
            # We need to denormalize them for filtering, then they get re-normalized in create_labels.
            # Let's adapt the utils functions or replicate logic here.
            
            # We assume the .npy file from process_and_create_video.py is already in the final YOLO format.
            # The old script did filtering on absolute pixel values, so we skip that for now,
            # as our boxes are already clipped and filtered in the previous step.
            
            labels = utils.create_labels(current_boxes) # Expects 'class_id', 'x', 'y', 'w', 'h'
            
            # Save this frame and its labels as a temporary H5 file
            # The fileList argument is used for naming, we'll use the file_index
            utils.save_compressed_clip(temp_dest_folder, f"{file_index}_{sub_sequence_idx}", split, "", [img_tensor.numpy()], [labels])
            sub_sequence_idx += 1

def merge_h5_files(temp_dir, final_dir, split):
    """Merges all temporary H5 files for a split into a single large H5 file."""
    print(f"\n--- Merging H5 files for {split} split ---")
    temp_split_dir = Path(temp_dir) / 'images' / split
    files = sorted(glob.glob(str(temp_split_dir / '*.h5')))

    if not files:
        print(f"No temporary H5 files found for {split} split. Skipping merge.")
        return

    final_h5_path = Path(final_dir) / f"{split}.h5"
    
    with h5py.File(files[0], 'r') as h_first:
        data = h_first['1mp']
        with h5py.File(final_h5_path, 'w') as hf:
            hf.create_dataset('1mp', data=data[:], chunks=True, maxshape=(None, data.shape[1], data.shape[2], data.shape[3]), **hdf5plugin.Blosc(cname='zstd'))
    os.remove(files[0])

    for f_path in tqdm(files[1:], desc=f"Merging {split} files"):
        with h5py.File(f_path, 'r') as h_add:
            data_to_add = h_add['1mp']
            with h5py.File(final_h5_path, 'a') as hf:
                current_size = hf['1mp'].shape[0]
                new_size = current_size + data_to_add.shape[0]
                hf['1mp'].resize(new_size, axis=0)
                hf['1mp'][current_size:] = data_to_add[:]
        os.remove(f_path)
    print(f"Successfully created {final_h5_path}")

def main():
    """Main function to run the final dataset processing."""
    print("--- Starting Final Dataset Preparation for ReYOLOv8 ---")
    
    # 1. Setup temporary directory for intermediate H5 files
    temp_output_dir = Path(FINAL_OUTPUT_PATH) / "temp"
    if temp_output_dir.exists():
        import shutil
        shutil.rmtree(temp_output_dir)

    for split in ["train", "validation"]:
        # 2. Process each split
        print(f"\n--- Processing {split} split ---")
        event_split_dir = Path(EVENT_DATA_PATH) / split
        label_split_dir = Path(LABEL_PATH) / split / "labels"
        temp_dest_folder = utils.create_destination_folder(str(temp_output_dir), {"method": METHOD, "timeWindow": TIME_WINDOW_MS, "tbin": TBIN}, "ReYOLOv8", IMG_WIDTH, IMG_HEIGHT)

        event_dirs = sorted([d for d in event_split_dir.iterdir() if d.is_dir()])
        if not event_dirs:
            print(f"错误: 在 {event_split_dir} 中没有找到事件数据目录。", file=sys.stderr)
            print("请先确保 v2e 步骤已成功运行。", file=sys.stderr)
            return

        for i, event_dir in enumerate(tqdm(event_dirs, desc=f"Processing {split} files")):
            base_name = event_dir.name
            event_file = event_dir / "events.aedat4"
            label_file = label_split_dir / f"{base_name}.npy"

            if event_file.exists() and label_file.exists():
                process_file_pair(event_file, label_file, temp_dest_folder, split, i)
            else:
                tqdm.write(f"警告: Skipping {base_name}, missing event or label file.")
        
        # 3. Merge H5 files for the split
        merge_h5_files(temp_dest_folder, FINAL_OUTPUT_PATH, split)

    print("\n--- All processing complete! ---")
    print(f"Final dataset is ready at: {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
