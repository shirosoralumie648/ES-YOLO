"Takes .dat and .npy files as inputs and produce as grauscale jpg images + YOLO Darknet txt annotation files"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
from prophesee.io.psee_loader import PSEELoader
import codecs
import time 
import os 
from utils import create_destination_folder, to_bbox_yolo_format,  filter_boxes, create_labels, save_compressed_clip, clip_boxes
import torch
import h5py
from scipy import sparse
import hdf5plugin



def arg_parse():
    parser = argparse.ArgumentParser(description='Get Information Necessary to convert Event Spikes to Dense Formats')
    parser.add_argument("--timeWindow", dest = 'timeWindow', help = "Time window used to aggregate events", default = 50, type = int)
    parser.add_argument("--dataset",dest='dataset',default="GEN1",help="Choose from: GEN1, 1MP_3classes, 1MP_7classes",type=str,choices=["GEN1", "1MP_3classes", "1MP_7classes"])    
    parser.add_argument("--category", dest = 'category', help = "train, test, or val", default = "train",type = str)
    parser.add_argument("--source", dest = 'source', help = "source of input files", default = '/home/silvada/Desktop/TestDatasetHandler/reduced_GEN1/raw', type = str)
    parser.add_argument("--destination", dest = 'destination', help = "source of input files", default = '/home/silvada/Desktop/TestDatasetHandler/reduced_GEN1/', type = str)
    parser.add_argument("--list", dest = 'list', help = "current file list", default = '00', type = str)
    parser.add_argument("--img_x", dest = 'size_x', help = "horizontal axis", default = 304, type = int)
    parser.add_argument("--img_y", dest = 'size_y', help = "vertical axis", default = 240, type = int)
    parser.add_argument("--name", dest = 'name', help = "name of the dataset", default = "GEN1", type = str)
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


downsample = (dataset != "GEN1")


filter_diag = 30 if dataset == "GEN1" else 60
filter_size = 10 if dataset == "GEN1" else 20

print(filter_diag)
print(filter_size)
print(downsample)

formatInfo = {"method": method, "tbin": tbin, "timeWindow": timeWindow}

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
if dataset == "GEN1":
  print(os.path.join(os.path.join(PATH, category)))
else:
 print(os.path.join(os.path.join(PATH,category + 'filelist' + fileList, category)))

index = 0
labels=None
if dataset == "GEN1":
 lines = glob.glob(((os.path.join(os.path.join(PATH, category)) + '/*.dat')))
else: 
 lines = glob.glob(((os.path.join(PATH,category + 'filelist' + fileList, category) + '/*.dat')))
num = len(lines)
start_time = time.time()
print("number of clips",num)


for k in range(num):


  print('********************** Current Iteration ******************************')
  print(k)
  current_time = time.time() - start_time
  print('*************** TIME ELAPSED ******************')
  print(current_time)
  td_file = lines[k]
  print('**************** MY INPUT FILE ***************')
  print(td_file)
  
  ##### Load Boxes and Labels from corresponding ".dat"/".npy" pairs
  if td_file.endswith('_td.dat'):
   box_file = td_file[:-7]
  
  
  # use the naming pattern to find the corresponding box file


  box_file = box_file + '_bbox.npy'
  videos = PSEELoader(td_file)
  box_videos = PSEELoader(box_file)

  
  # Filter out the first 0.5s from each recording. 
  # TO DO: Make it optional at the arg_parse
  events = videos.load_delta_t(tstart)
  box_events = box_videos.load_delta_t(tstart)
  
  ### Iterate through the videos....
  imgs = []
  imgs_c = []
  labels = []

  while not videos.done:
      # load events and boxes from all files
      

      events = videos.load_delta_t(delta_t)
      box_events = box_videos.load_delta_t(delta_t)

      if not (events.size == 0):   

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

        #img = img.numpy()
        #img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
        
        # clip and filter boxes
        box_events = clip_boxes(boxes = box_events, width = width, height = height)
        box_events = filter_boxes(box_events,0,filter_diag,filter_size,dataset)

        #img = torch.cat(list(img), dim = 0) 
        #img = img.to("cpu").numpy().astype(np.int8)
        box_not_empty = (box_events.size != 0)
        
        #print("box empty", box_events.size)

        if (box_not_empty):
           # convert bboxes to yolo format
           box_events = to_bbox_yolo_format(box_events = box_events, width = width, height = height)
           imgs.append(img)
           labels.append(create_labels(box_events))
               
        index += 1
  if len(imgs) != 0:
   save_compressed_clip(destFolder, k, category, fileList, imgs, labels, True)
   
   
   

##################################### STEP 2: MERGE H5 FILES ###################################################   
   
original_path = os.path.join(destFolder, "images", category)
dest_folder = os.path.join(destFolder, "images", category)
if method in ["vtei", "mdes"]:
   ch = bins
else: 
    ch = 2*bins


name = name + "_" + category + ".h5"

files = sorted(glob.glob(os.path.join(original_path,'*.h5')))
print(files)
#for i in range(len(files)):
#print(files[i])
#print(files[0])

with h5py.File(files[0], 'r') as h:
 #h = h5py.File(files[0],'r')

 w = h['1mp']
 #w = w[:,:,:,:].transpose((0,1,3,2))
 print(w.shape)
 old_len = len(w)
 print(h.keys())
 hf = h5py.File(os.path.join(dest_folder,name),'w')

 hf.create_dataset('1mp', data=w, chunks=True, maxshape=(None,ch,height, width), **hdf5plugin.Blosc(cname='zstd')) 

os.remove(files[0])

for i in range(1,len(files)):

    with h5py.File(files[i], 'r') as h:
     w = h['1mp']
     print(files[i],len(w))
     #if len(w) < 11:
     # print(len(w))
     #w = w[:,:,:,:].transpose((0,1,3,2))
     new_len = len(w)
    
     hf['1mp'].resize(hf['1mp'].shape[0] + new_len, axis = 0)
     hf['1mp'][-new_len:] = w
    os.remove(files[i])

print(len(hf['1mp']))
hf.close()   
   
   

   

