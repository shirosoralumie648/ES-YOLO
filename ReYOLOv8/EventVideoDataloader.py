import torch
import numpy as np
import os
import random
import sys 
import time
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.utils import  PIN_MEMORY, RANK
from EventVideoDataset import EventVideoDetectionDataset
from torch.utils.data import DataLoader, dataloader, distributed
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first



def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_video_dataloader(cfg, video_config, batch_size, video_path, aug_param, mode, rank=-1, load = "batched", random_seed = False):

    shuffle = (mode == "train")
    #print("video path", video_path)
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = EventVideoDetectionDataset(video_path,video_config["clip_length"], video_config["clip_stride"], video_config["channels"], aug_param,mode, load)

    batch_size = min(batch_size, len(dataset))
  
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == "train" else cfg.workers * 2
    #workers = cfg
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    #nw = workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader # allow attribute updates
    generator = torch.Generator()
    if not random_seed:
     generator.manual_seed(6148914691236517205  + RANK)
    
    return loader(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, "collate_fn", None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def build_video_val_standalone_dataloader(cfg, video_config, batch_size, video_path, rank=-1, mode = "sequential", speed = False, zero_hidden = False):

    shuffle = False 
    batch_size = 1    

    if mode != "sequential":
       batch_size = batch_size

    if zero_hidden:  

       mode = "batched"
    
    if speed:  

       video_config["clip_length"] = 1
       video_config["clip_stride"] = 1
       mode = "batched"  

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        
        dataset = EventVideoDetectionDataset(video_path,video_config["clip_length"], video_config["clip_stride"], video_config["channels"], [None],"val", mode)

  
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == "train" else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    
    return loader(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, "collate_fn_val", None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


