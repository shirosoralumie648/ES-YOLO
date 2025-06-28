from __future__ import print_function
import numpy as np
import torch
import math
import time


def shist(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.uint8
    #assert p.min() >= 0
    #assert p.max() <= 1
    representation = torch.zeros((2,bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)

    indices = x.long() + \
                  width * y.long() + \
                  height *  width  * t_idx.long() + \
                  bins * height * width * p.long()
    values = torch.ones_like(indices, dtype=dtype, device=device)
    representation.put_(indices, values, accumulate=True)
    representation = torch.clamp(representation, min=0, max=255)

    return torch.reshape(representation, (-1, height, width))


def voxel_grid(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.half
    #assert p.min() >= 0
    #assert p.max() <= 1

    
    representation = torch.zeros((2,bins, height, width), dtype=dtype)
    t0 = t[0]
    t1 = t[-1]
   
    tnorm = t - t0
    
    tnorm = tnorm/max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    values = torch.maximum(torch.zeros_like(tnorm, dtype= dtype), 1 - torch.abs(tnorm - t_idx)).to(dtype=dtype)

    

    indices = x.long() + \
                  width * y.long() + \
                  height *  width  * t_idx.long() + \
                  bins * height * width * p.long()
    
    representation.put_(indices, values, accumulate=True)
    return torch.reshape(representation, (-1, height, width))

def ev_temporal_volume(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int16
    #assert p.min() >= 0
    #assert p.max() <= 1
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    p = 2*p - 1

    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    
    indices = x.long() + width*y.long() + height*width*t_idx.long()

   

    values = torch.asarray(p, dtype=dtype, device = device)
    
    
    
    representation.put_(indices, values, accumulate=True)

    
    #representation = torch.clamp(representation, min=-1, max=1)
   
    return torch.reshape((255.0/( 1 + torch.exp(-representation/2))).to(dtype=torch.uint8), (-1, height, width))


def vtei(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int8
    #assert p.min() >= 0
    #assert p.max() <= 1

    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    t0 = t[0]
    t1 = t[-1]
    
    p = 2*p - 1

    tnorm = t - t0
    tnorm = tnorm/ max((t1-t0),1)
    tnorm = tnorm*bins
    t_idx = tnorm.floor()
    t_idx = torch.clamp(t_idx, max = bins - 1)
    
    indices = x.long() + width*y.long() + height*width*t_idx.long()

   

    values = torch.asarray(p, dtype=dtype, device = device)
    
    
    
    representation.put_(indices, values, accumulate=False)

    
    #representation = torch.clamp(representation, min=-1, max=1)
   
    return torch.reshape(representation, (-1, height, width))

def mdes(x,y,t,p, bins, height, width, device = "cpu"):
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/representations.py#L124
    dtype = torch.int8
    representation = torch.zeros((bins, height, width), dtype=dtype, device=device, requires_grad=False)
    
    p = 2*p - 1

    t0 = t[0]
    t1 = t[-1]
    tnorm = (t - t0)/ max((t1-t0),1)
    tnorm = torch.clamp(tnorm, min=1e-6, max = 1 - 1e-6)
    bin_float = bins - torch.log(tnorm)/ math.log(1/2)
    bin_float = torch.clamp(bin_float, min = 0)
    t_idx = bin_float.floor()
    indices = x.long() + width*y.long() + height*width*t_idx.long()
    
    values = torch.asarray(p, dtype=dtype, device = device)
    representation.put_(indices, values, accumulate=True)
    
    

    for i in reversed(range(bins)):
        representation[i] = torch.sum(input=representation[:i + 1], dim=0)


    return representation

