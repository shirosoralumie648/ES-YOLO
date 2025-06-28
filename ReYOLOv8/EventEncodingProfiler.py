import csv
import math
import numpy as np
import argparse
from os import listdir
import glob
import array as arr
import time 
import os 
import torch
import time
import pandas as pd
from pathlib import Path



parser = argparse.ArgumentParser(description="Process a folder string argument.")
parser.add_argument('--folder', type=str, help='The path to the folder')
args = parser.parse_args()
folder = args.folder

formats = ["vtei", "shist", "mdes", "voxel_grid"]
fmt = "*.csv"

ts = 50

fmt_ls = []
latency_ls = []
sparsity_ls = []
events_processed_ls = [] 
event_rate_ls = []
encoded_size_ls = []
compression_ratio_ls = []
bw_ls = []
for fmt2 in formats:
 file_list = glob.glob(os.path.join(folder, fmt2, fmt))
 target_file = file_list[0]

 latency = []
 events_processed =[]
 sparsity = []
 data = pd.read_csv(target_file)
 latency.append(data["time_elapsed"])
 events_processed.append(data["number of processed events"])
 sparsity.append( data["non-zeros "])
 fmt_ls.append(fmt2)
 latency_ls.append(1e3*np.mean(latency))
 sparsity_ls.append(np.mean(sparsity))
 events_processed_ls.append(np.mean(events_processed))
 event_rate_ls.append(np.mean(events_processed) / (np.mean(latency))/(1e6))
 
 if fmt2 in {"vtei", "mdes"}:
     encoded_size_ls.append(3*np.mean(sparsity)/(1024**2))
     
     compression_ratio_ls.append((4*np.mean(events_processed))/(3*np.mean(sparsity)))
     bw_ls.append((3*np.mean(sparsity)/(1024**2)) / (1e-3*(ts + 1e3*np.mean(latency))))
     
 elif fmt2 == "shist":
     encoded_size_ls.append(4*np.mean(sparsity)/(1024**2))
     bw_ls.append((4*np.mean(sparsity)/(1024**2)) / (1e-3*(ts + 1e3*np.mean(latency))))
     compression_ratio_ls.append((4*np.mean(events_processed))/(4*np.mean(sparsity)))
 else:
     encoded_size_ls.append(5*np.mean(sparsity)/(1024**2))
     bw_ls.append((5*np.mean(sparsity)/(1024**2)) / (1e-3*(ts + 1e3*np.mean(latency))))
     compression_ratio_ls.append((4*np.mean(events_processed))/(5*np.mean(sparsity)))
 
df_data = {"formats" : fmt_ls, "latency @50ms (ms)": latency_ls, "event rate (Mev/s)": event_rate_ls,"non-zeros": sparsity_ls, "Encoded Size ": encoded_size_ls, "Compression Ratio": compression_ratio_ls,  "BW (MB/s)": bw_ls,"events_processed": events_processed_ls}
df = pd.DataFrame(df_data)

max_ =  np.max(events_processed)
idx = (events_processed[0].tolist()).index(max_)


latency_ls2 = []
events_processed_ls2 =[]
sparsity_ls2 = []
event_rate_ls2 = []
encoded_size_ls2 = []
compression_ratio_ls2 = []
bw_ls2 = []
for fmt2 in formats:
 file_list = glob.glob(os.path.join(folder, fmt2, fmt))
 target_file = file_list[0]
 data = pd.read_csv(target_file)


 sparsity_ls2.append(data["non-zeros "][idx])
 latency_ls2.append(1e3*data["time_elapsed"][idx])
 events_processed_ls2.append(data["number of processed events"][idx])
 event_rate_ls2.append(data["number of processed events"][idx] / data["time_elapsed"][idx]/(1e6))
 
  
 if fmt2 in {"vtei", "mdes"}:
     encoded_size_ls2.append(3*data["non-zeros "][idx]/(1024**2))
     bw_ls2.append((3*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls2.append((4*data["number of processed events"][idx])/(3*data["non-zeros "][idx]))
 elif fmt2 == "shist":
     encoded_size_ls2.append(4*data["non-zeros "][idx]/(1024**2))
     bw_ls2.append((4*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls2.append((4*data["number of processed events"][idx])/(4*data["non-zeros "][idx]))
 else:
     encoded_size_ls2.append(5*data["non-zeros "][idx]/(1024**2))
     bw_ls2.append((5*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls2.append((4*data["number of processed events"][idx])/(5*data["non-zeros "][idx]))
 
df_data = {"formats" : fmt_ls, "latency @50ms (ms)": latency_ls2, "event rate (Mev/s)": event_rate_ls2,"non-zeros": sparsity_ls2, "Encoded Size ": encoded_size_ls2, "Compression Ratio": compression_ratio_ls2, "BW (MB/s)": bw_ls2,"events_processed": events_processed_ls2}
df2 = pd.DataFrame(df_data)


min_ =  np.min(events_processed)
idx = (events_processed[0].tolist()).index(min_)


latency_ls3 = []
events_processed_ls3 =[]
sparsity_ls3 = []
event_rate_ls3 = []
encoded_size_ls3 = []
compression_ratio_ls3 = []
bw_ls3 = []
for fmt2 in formats:
 file_list = glob.glob(os.path.join(folder, fmt2, fmt))
 target_file = file_list[0]
 data = pd.read_csv(target_file)


 sparsity_ls3.append(data["non-zeros "][idx])
 latency_ls3.append(1e3*data["time_elapsed"][idx])
 events_processed_ls3.append(data["number of processed events"][idx])
 event_rate_ls3.append(data["number of processed events"][idx] / data["time_elapsed"][idx]/(1e6))
 if fmt2 in {"vtei", "mdes"}:
     encoded_size_ls3.append(3*data["non-zeros "][idx]/(1024**2))
     bw_ls3.append((3*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls3.append((4*data["number of processed events"][idx])/(3*data["non-zeros "][idx]))
 elif fmt2 == "shist":
     encoded_size_ls3.append(4*data["non-zeros "][idx]/(1024**2))
     bw_ls3.append((4*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls3.append((4*data["number of processed events"][idx])/(4*data["non-zeros "][idx]))
 else:
     encoded_size_ls3.append(5*data["non-zeros "][idx]/(1024**2))
     bw_ls3.append((5*data["non-zeros "][idx]/(1024**2)) / (1e-3*(ts + 1e3*data["time_elapsed"][idx])))
     compression_ratio_ls3.append((4*data["number of processed events"][idx])/(5*data["non-zeros "][idx]))
     
     
df_data = {"formats" : fmt_ls, "latency @50ms (ms)": latency_ls3, "event rate (Mev/s)": event_rate_ls3,"non-zeros": sparsity_ls3, "Encoded Size ": encoded_size_ls3,"Compression Ratio": compression_ratio_ls3,"BW (MB/s)": bw_ls3,"events_processed": events_processed_ls3}
df3 = pd.DataFrame(df_data)

print(" Scenario 1, Average #Events Processed:", int(np.mean(events_processed_ls)), "Disk Size (MB):", int(np.mean(events_processed_ls))*4/(1024**2))
print(df)
print(" Scenario 2, Maximum #Events Processed:", int(np.mean(events_processed_ls2)), "Disk Size (MB):", int(np.mean(events_processed_ls2))*4/(1024**2))
print(df2)
print(" Scenario 3, Minimum #Events Processed:", int(np.mean(events_processed_ls3)), "Disk Size (MB):", int(np.mean(events_processed_ls3))*4/(1024**2))
print(df3)
