"Takes .npy and .txt files as inputs and produces h5-based data"
import csv
import math
import numpy as np
import cv2
import argparse
import csv
from os import listdir
from os import path, environ
from os.path import isfile, join
import glob
import array as arr
import formats_utils as fmt
import codecs
import time 
import os 
from utils import create_destination_folder, to_bbox_yolo_format,  filter_boxes, create_labels, save_compressed_clip, save_only_compressed_clip, clip_boxes
import torch
import h5py
from scipy import sparse
import hdf5plugin
import json


def arg_parse():
    parser = argparse.ArgumentParser(description='Get Information Necessary to convert Event Spikes to Dense Formats')
    parser.add_argument("--timeWindow", dest = 'timeWindow', help = "Time window used to aggregate events", default = 40, type = int)
    parser.add_argument("--dataset",dest='dataset',default="pedro",help="pedro dataset",type=str,choices=["pedro"])    
    parser.add_argument("--category", dest = 'category', help = "train, test, or val", default = "train",type = str)
    parser.add_argument("--source", dest = 'source', help = "source of input files", default = '/home/silvada/Desktop/TestDatasetHandler/pedro/numpy', type = str)
    parser.add_argument("--destination", dest = 'destination', help = "source of input files", default = '/home/silvada/Desktop/TestDatasetHandler/pedro/', type = str)
    parser.add_argument("--list", dest = 'list', help = "current file list", default = '00', type = str)
    parser.add_argument("--img_x", dest = 'size_x', help = "horizontal axis", default = 346, type = int)
    parser.add_argument("--img_y", dest = 'size_y', help = "vertical axis", default = 260, type = int)
    parser.add_argument("--name", dest = 'name', help = "name of the dataset", default = "pedro", type = str)
    parser.add_argument("--device", dest = 'device', help = "device to perform operations", default = "cpu", type = str)
    parser.add_argument("--method",dest='method',default="vtei",help="Choose from: vtei, mdes, voxel_grid, shist, ergo12",type=str,choices=["vtei", "ergo12", "mdes", "voxel_grid", "shist", "taf"])    
    parser.add_argument("--bins", dest = 'tbin', help = "number of bins for voxel grids", default = 10, type = int)
    return parser.parse_args()



################################## 1ST STEP: GENERATE THE INDIVIDUAL RECORDINGS ######################################

# Unrolling parse arguments
args = arg_parse()

timeWindow = args.timeWindow
name = args.name
PATH = args.source
category = args.category
destination = args.destination
fileList = args.list
dataset = args.dataset
size_y = args.size_y
size_x = args.size_x
method = args.method
tbin = args.tbin
device = args.device


formatInfo = {"method": method, "tbin": tbin, "timeWindow": timeWindow}

print(f"format info {formatInfo}")
# Parameter Initialization
destFolder = create_destination_folder(destination,formatInfo,name,size_x,size_y)

delta_t= 1000*timeWindow # for milioseconds as input
skip_ts= 0 

width = size_x
height = size_y
weight = 10
tstart = 500000    

bins = tbin
# File Initialization
# TO DO: USE TQDM AND LOGGER TO TURN THIS TERMINAL LOGGING MORE FANCY
print('dest folder')
print(destFolder)
print('source')
print(os.path.join(os.path.join(PATH, category)))

index = 0
labels=None
lines = sorted(glob.glob(((os.path.join(os.path.join(PATH, category)) + '/*.npy'))))

num = len(lines)
start_time = time.time()
print("number of clips",num)


######### STEP 1: CONVERT THE RAW NUMPY FILES TO H5 FILES ############### 
for k in range(num):


  imgs = []
  print('********************** Current Iteration ******************************')
  print(k)
  current_time = time.time() - start_time
  print('*************** TIME ELAPSED ******************')
  print(current_time)
  td_file = lines[k]
  print('**************** MY INPUT FILE ***************')
  print(td_file)
  
  events = np.load(td_file)
  events = np.array([(events[i,0],events[i,1],events[i,2],events[i,3]) for i in range(len(events))], dtype = [('t','f8'), ('x','i4'), ('y','i4'), ('p','i1')])
  boxes = np.loadtxt(str.replace(str.replace(td_file, "numpy", "yolo"),".npy",".txt"))
  if not (events.size == 0) and boxes.size != 0:   

        x = torch.from_numpy(np.clip(events["x"].astype(np.int64).copy(),0,width-1))
        y = torch.from_numpy(np.clip(events["y"].astype(np.int64).copy(),0,height-1))
        t = torch.from_numpy(events["t"].astype(np.int64))
        p  = torch.from_numpy(events["p"].astype(np.int64).copy())
        num_events = len(x)
        if method == "vtei":
         img = fmt.vtei(x, y, t, p, bins, height, width).numpy()  
        elif method == "shist":
         img = fmt.shist(x, y, t, p, bins, height, width).numpy()  
        elif method == "voxel_grid":
         img = fmt.voxel_grid(x, y, t, p, bins, height, width).numpy()  
        elif method == "ergo12":
         img = fmt.ergo(x,y,t,p, num_events, height, width)
         img = np.transpose(img, (2,0,1))

        else:
         img = fmt.mdes(x, y, t, p, bins, height, width).numpy()   

        imgs.append(img)

        index += 1
        if len(imgs) != 0:
           save_only_compressed_clip(destFolder, index, category, fileList, img, compress = True)
   
   
##################################### STEP 2: Generate the labels according to  #######################################

pedro_sequence_metadata = os.path.join(os.getcwd(),"pedro_metadata/pedro_sequences.json")
label_dir = sorted(glob.glob(os.path.join(str.replace(PATH, "numpy", "yolo"), category, "*.txt")))

with open(pedro_sequence_metadata, 'r') as file:
    data = json.load(file)

list_name  = list(data[category].keys())
idx_begin = [data[category][i]["begin"] for i in list_name]
idx_end = [data[category][i]["end"] for i in list_name]
ctr = 0

for i in range(len(list_name)):
    labels = []
    files_ = []
    for j in range(int(idx_begin[i]),int(idx_end[i] + 1)):

        cur_labels = np.loadtxt(label_dir[j])
        files_.append(label_dir[j])
        if cur_labels.size != 0: 
         if cur_labels.ndim == 1:
          cur_labels = cur_labels.reshape(1,5)
        

         label = np.zeros((cur_labels.shape[0],5))
    
         label[:,0] = cur_labels[:,0]
         label[:,1] = cur_labels[:,1]
         label[:,2] = cur_labels[:,2]
         label[:,3] = cur_labels[:,3]
         label[:,4] = cur_labels[:,4]
         labels.append(label) 
        else: 
         print(f"Empty file {label_dir[ctr]}")
    
    print(np.array(labels, dtype = object).shape, idx_begin[i], idx_end[i])

    np.save(os.path.join(destFolder + '/labels/'+ category+'/sequence_' + fileList + '_subseq_' + str(i).zfill(7) +'.npy'),np.array(labels, dtype = object))  
##################################### STEP 3: MERGE H5 FILES ###################################################   
   
original_path = os.path.join(destFolder, "images", category)
dest_folder = os.path.join(destFolder, "images", category)
if method in ["vtei", "mdes"]:
   ch = bins
else: 
    ch = 2*bins


name = name + "_" + category + ".h5"

files = sorted(glob.glob(os.path.join(original_path,'*.h5')))
print(files, len(files))
#for i in range(len(files)):
#print(files[i])
#print(files[0])


with h5py.File(files[0], 'r') as h:
 #h = h5py.File(files[0],'r')

 w = h['1mp']
 #w = w[:,:,:,:].transpose((0,1,3,2))
 w = w[:,:,:].reshape(1,*w.shape)
 old_len = len(w)
 hf = h5py.File(os.path.join(dest_folder,name),'w')

 hf.create_dataset('1mp', data=w, chunks=True, maxshape=(None,ch,height, width), **hdf5plugin.Blosc(cname='zstd')) 

os.remove(files[0])

for i in range(1,len(files)):

    with h5py.File(files[i], 'r') as h:
     w = h['1mp']
     w = w[:,:,:].reshape(1,*w.shape)
     print(files[i])
     #if len(w) < 11:
     # print(len(w))
     #w = w[:,:,:,:].transpose((0,1,3,2))
     new_len = len(w)
    
     hf['1mp'].resize(hf['1mp'].shape[0] + new_len, axis = 0)
     hf['1mp'][-new_len:] = w
    os.remove(files[i])

print(len(hf['1mp']))
hf.close()   
