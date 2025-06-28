from os import path, environ
from os.path import isfile, join
import glob
import array as arr
import formats_utils as fmts
from prophesee.io.psee_loader import PSEELoader
import codecs
import time 
import os 
import torch
import time
import sys 
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Process input and output folders and a format option.")


parser.add_argument('--input_file', type=str, help='Event sequence file to be processed')
parser.add_argument('--output', type=str, help='The path to the output folder', default="test_timing")
parser.add_argument('--format', type=str, choices=['vtei', 'voxel_grid', 'shist', 'mdes'],
                        help='event encodings', required=True)
parser.add_argument('--bins', type=int, default=5, help='The number of bins to use')

args = parser.parse_args()


input_folder = args.input_file
output_folder = args.output
format_ = args.format
bins = args.bins

if format_ in ["shist","voxel_grid"]:
   bins = 2*bins
   
timeWindow = 50
delta_t= 1000*timeWindow

skip_ts= 0 

width = 304
height = 240
weight = 10
tstart = 500000    


td_file = os.path.join(os.getcwd(),input_folder)


t = 1
while t == 1:

 videos = PSEELoader(td_file)

 number_of_events = []
 storage = []
 time_elapsed = []
 non_zeros = []

 events = videos.load_delta_t(tstart)



 while not videos.done:
  events = videos.load_delta_t(delta_t)
  img = []


  if not (events.size == 0): 
   
  

    x = torch.from_numpy(np.clip(events["x"].astype(np.int64).copy(),0,width-1))
    y = torch.from_numpy(np.clip(events["y"].astype(np.int64).copy(),0,height-1))
    t = torch.from_numpy(events["t"].astype(np.int64))
    p = torch.from_numpy(events["p"].astype(np.int64).copy())

    start_time = time.time()
    
    
    if format_ == "mdes":
     img = fmts.mdes(x, y, t, p, timeWindow//bins, height, width)
    elif format_ == "shist":
     img = fmts.shist(x, y, t, p, timeWindow//bins, height, width)
    elif format_ == "vtei":
     img = fmts.vtei(x, y, t, p, timeWindow//bins, height, width)
    elif format_ == "voxel_grid":
     img = fmts.voxel_grid(x, y, t, p, timeWindow//bins, height, width)
    else:

     sys.exit(0)
        
    
    time_elapsed.append(time.time() - start_time)
    number_of_events.append(len(events))
    storage.append(sys.getsizeof(events))
    non_zeros.append(np.count_nonzero(img))
  
  data = {"time_elapsed" : time_elapsed, "number of processed events" : number_of_events, "event storage " : storage, "non-zeros ": non_zeros}

 df = pd.DataFrame(data)
 print(df)
 t = 0
 os.makedirs(os.path.join(output_folder, format_), exist_ok=True)
 df.to_csv(output_folder + "/"+ format_ + "/"+ Path(td_file).stem + ".csv")
 print("time_elapsed :", np.mean(time_elapsed), "number of events :", np.mean(number_of_events), "event storage :", np.mean(storage), "sparsity :", 1 -  np.mean(non_zeros)/(5*304*240))

  
