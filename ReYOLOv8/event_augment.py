import numpy as np
import random
from torch.nn.functional import interpolate
import torch

class RandomFlip:
    # https://github.com/uzh-rpg/RVT/blob/master/data/utils/augmentor.py

    def __init__(self, p=0.5, direction='horizontal') -> None:
        assert direction in ['horizontal'], f'Support direction `horizontal`, got {direction}'
        assert 0 <= p <= 1.0
        self.p = p
        self.direction = direction

    def __call__(self, images, boxes):
        img = images
        instances = boxes
        # Flip right-left
        if self.direction == 'horizontal' and random.random() < self.p:
            img = np.flip(img,axis=-1)
            instances[:,0] = abs(1 -  instances[:,0])
            # flip the boxes 
            
            #instances.fliplr(w)
        images = np.ascontiguousarray(img)
        boxes = instances
        return images, boxes
        
    
class ZoomOut: 
      # https://github.com/uzh-rpg/RVT/blob/master/data/utils/augmentor.py
      def __init__(self, p = 0.5, min_zoom_out_factor = 1.0, max_zoom_out_factor = 1.2):
        
        self.p = p
        self.min_zoom_out_factor = min_zoom_out_factor
        self.max_zoom_out_factor = max_zoom_out_factor
      
      def __call__(self, images, boxes):
        img = images
        instances = boxes
        if random.random() < self.p:
         l, c, h, w = img.shape
        
         rand_zoom_out_factor = torch.distributions.Uniform(
         self.min_zoom_out_factor, self.max_zoom_out_factor).sample()
         zoom_window_h, zoom_window_w = int(h / rand_zoom_out_factor), int(w / rand_zoom_out_factor)
         x0_sampled = int(torch.randint(low=0, high=w - zoom_window_w,size = (1,)))
         y0_sampled = int(torch.randint(low=0, high=h - zoom_window_h,size = (1,)))
         zoom_window = interpolate(torch.from_numpy(np.array(img, dtype = np.float32)), size=(zoom_window_h, zoom_window_w), mode='nearest-exact')
         images = torch.zeros_like(torch.from_numpy(np.array(img, dtype = np.float32)))
         images[:,:, y0_sampled:y0_sampled + zoom_window_h, x0_sampled:x0_sampled + zoom_window_w] = zoom_window
         instances[:,0] = (zoom_window_w*instances[:,0] + x0_sampled)/w
         instances[:,1] = (zoom_window_h*instances[:,1] + y0_sampled)/h
         instances[:,2] = instances[:,2]*(w//zoom_window_w)
         instances[:,3] =instances[:,3]*(h//zoom_window_h)
         images = images.numpy()
        else:
         images = np.ascontiguousarray(img)
        
        boxes = instances
        return images, boxes
         
      
class InvertPolarity:

    def __init__(self, p=0.5) -> None:

        assert 0 <= p <= 1.0

        self.p = p


    def __call__(self, images, boxes):
        img = images
        instances = boxes
        if random.random() < self.p:
            img = img*-1

        images = np.ascontiguousarray(img)
        boxes = instances
        return images, boxes
        

class SuppressPolarity:

    def __init__(self, p1=0.5, p2=0.5) -> None:

        assert 0 <= p1 <= 1.0
        assert 0 <= p2 <= 1.0

        self.p1 = p1
        self.p2 = p2 

    def __call__(self, images, boxes):
        img = images
        instances = boxes
        if  random.random() < self.p1:
         if random.random() < self.p2:
             img[img > 0] = 0
         else:
            img[img < 0] = 0

        images = np.ascontiguousarray(img)
        boxes = instances
        return images, boxes
        
        
class ApplyEventAugmentation:

    def __init__(self, aug_params) -> None:
        
        self.aug_params = aug_params 
        self.flip = self.aug_params["flip"]
        self.suppress = self.aug_params["suppress"]
        self.positive = self.aug_params["positive"]
        self.invert = self.aug_params["invert"]
        self.zoom_out = self.aug_params["zoom_out"]
        self.max_zoom_out_factor = self.aug_params["max_zoom_out_factor"]
        self.min_zoom_out_factor = self.aug_params["min_zoom_out_factor"]
        
        
        self.random_flip = RandomFlip(self.flip)
        self.suppress_polarity = SuppressPolarity(self.suppress, self.positive)
        self.invert_polarity = InvertPolarity(self.invert)
        self.zoom_out_aug = ZoomOut(self.zoom_out, self.min_zoom_out_factor, self.max_zoom_out_factor)
        
    def __call__(self, images, boxes):
        images, boxes = self.random_flip(images, boxes)
        images, boxes = self.suppress_polarity(images, boxes)
        images, boxes = self.invert_polarity(images,boxes)
        return self.zoom_out_aug(images, boxes)
 
 
