from torch.utils.data import Dataset
import sys
import os
import h5py 
import hdf5plugin
import numpy as np
import torch
import time 
import glob
import math 
from event_augment import ApplyEventAugmentation, ZoomOut


class EventVideoDetectionDataset(Dataset):
    # Based on:  https://github.com/MichiganCOG/ViP/blob/74776f2575bd5339ba39c784bbda4f04cc859add/datasets/abstract_datasets.py #
    
    def __init__(self,video_path, clip_length,clip_stride, channels, aug_param,load_type = "train", mode = "batched"):
        
        self.aug_param = aug_param
        self.video_path = video_path
        self.clip_length = clip_length
        self.clip_stride = clip_stride 
        self.mode = mode
        self.video_path_h5 = None
        self.data_file = None
        self.load_type = load_type
        self.sequence_last_clip = []
        self.sequence_length = []
        self.channels = channels
        if self.load_type == "train":
         self.transform = ApplyEventAugmentation(self.aug_param)
        self._getClips()

    def _getClips(self):

        self.samples = []
 
        self.video_path_h5 = glob.glob(os.path.join(self.video_path,'*.h5'))           
        self.label_files = sorted(glob.glob(os.path.join(self.video_path.replace('images','labels'),'*.npy')))
        
        # Load the information for each video and process it into clips
        begin = 0

        for i, f in enumerate(self.label_files):
            labels = np.load(f, allow_pickle = True)
            self.sequence_length.append(len(labels))
            length = len(labels)
             
            # padding is utilized to ensure clips of the same length
            labels = self.pad_labels(labels, length)
            indexes = self.pad_clip(length)

            video_info = [{"frame": begin + indexes[idx], "labels": labels[idx], "label_file": f} for idx in range(len(labels))]
            
            del labels


            clips= self._extractClips(video_info,length)
            begin = length + begin
            
            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:    
        
                self.samples.append(clip)
             
    def pad_labels(self, labels, length):

        if length < self.clip_length and self.mode == "batched":
           
           difference = self.clip_length - length  
           
           labels = list(labels)

           for i in range(difference):

               labels.append(labels[int(math.fmod(i, length))])
           
           

           return np.array(labels)    
           
        else:
             return labels  
    
    def pad_clip(self, length):
        

        
        difference = self.clip_length - length
        
        indexes = [i for i in range(length)]

        if length < self.clip_length and self.mode == "batched":
           for i in range(difference):

               indexes.append(indexes[int(math.fmod(i, length))])
           return indexes
        else:
             return indexes         

    def _extractClips(self,video,length):
        """
        Processes a single video into uniform sized clips that will be loaded by __getitem__
        Args:
            video: List containing a dictionary of annontations per frame
            length: lenght of each clip                              
             
        returns: final_video -> list with starting indexes for each clip
        """
        if self.load_type == "train" or self.mode == "batched":

          self.num_clips = len(video) // self.clip_stride
          required_length = (self.num_clips-1)*(self.clip_stride) + self.clip_length
          indices = np.tile(np.arange(0, required_length, 1, dtype='int32'),1)
          # Starting index of each clip
          clip_starts = np.arange(0, len(indices), self.clip_stride).astype('int32')[:self.num_clips]
        
                    
        
          if len(video) > required_length:
          
            clip_starts=np.append(clip_starts,len(video) - self.clip_length)
            self.num_clips += 1
          if self.sequence_last_clip == []:
             self.sequence_last_clip.append(self.num_clips - 1)
          else:

             self.sequence_last_clip.append(self.sequence_last_clip[len(self.sequence_last_clip) - 1] + self.num_clips)
            
          #print(clip_starts)
          final_video = [video[_idx:_idx+self.clip_length] for _idx in clip_starts]
        else:
             self.num_clips = 1
             self.clip_stride = 1
             self.clip_length = len(video)
 
             if self.sequence_last_clip == []:
              self.sequence_last_clip.append(self.num_clips - 1)
             else:

              self.sequence_last_clip.append(self.sequence_last_clip[len(self.sequence_last_clip) - 1] + self.num_clips)
             final_video = [video[0:self.clip_length]]
        
        return final_video
   
    def __len__(self):

       return len(self.samples)

    def __getitem__(self,idx):
        
        if not(self.data_file):
           self.data_file = h5py.File(self.video_path_h5[0],'r')
        
        vid_info = self.samples[idx]
        vid_length = len(vid_info)

        frame = [vid_info[idx2]['frame'] for idx2 in range(len(vid_info))]
        classes = [vid_info[idx3]['labels'][:,0] for idx3 in range(len(vid_info))]
        boxes = [vid_info[idx4]['labels'][:,1:] for idx4 in range(len(vid_info))]
        frame_ind = [[idx5]*len(vid_info[idx5]['labels']) for idx5 in range(len(vid_info))] 
        flat_frame_ind = [item for sublist in frame_ind for item in sublist]
        cur_label = np.array([self.data_file['1mp'][s,:,:,:] for s in frame]) #for idx6 in range(len(vid_info))]
        sequence_data = [idx]*(len(vid_info))
        img_file = [self.video_path_h5]*(len(vid_info))
        label_file = [vid_info[idx7]['label_file'] for idx7 in range(len(vid_info))]
        

        
         
        ret_dict = {}        
        boxes = np.concatenate(boxes, axis = 0, dtype = np.float64, casting = "unsafe")
        ### Apply The Augmentation
        if self.load_type == "train":
         cur_label, boxes = self.transform(cur_label,boxes)
        
        ### Create the output dictionaries
        ret_dict['img']  = torch.from_numpy(cur_label.copy())
        ret_dict['bboxes'] = torch.from_numpy(boxes)
        ret_dict['cls'] = torch.from_numpy(np.concatenate(classes, axis = 0, dtype = np.float64, casting = "unsafe"))
        ret_dict['sequence'] =  sequence_data
        ret_dict['vid_file'] = img_file
        ret_dict['vid_pos'] = torch.from_numpy(np.array(flat_frame_ind))
        ret_dict['clip_pos'] = torch.from_numpy(np.array(frame))
        ret_dict['batch_idx'] = torch.zeros(ret_dict['cls'].shape)        

        return ret_dict
             

    @staticmethod
    def collate_fn_val(batch):

        # similar to the Ultralytics one
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        


        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                # make copies of tensors to match the size of a big tensor
                fill = torch.Tensor([value[0].shape[0] / value[i].shape[0]  for i in range(0,len(value))])
                
                fill_mask = fill > 2

                if fill_mask.any():      

                   idx = np.where(fill_mask)[0][0]    

                   value = list(value)     
                   value[idx:] = [torch.cat([value[idx+k1]]*int(fill[idx+k1]),0) for k1 in range(0,fill_mask.sum())]
                   value = tuple(value)

                   values[5] = list(values[5])
                   values[5][idx:] = [torch.cat([values[5][idx + k1]]*int(fill[idx+k1]),0) for k1 in range(0,fill_mask.sum())]
                   values[5] = tuple(values[5])

                   values[1] = list(values[1])
                   values[1][idx:] = [torch.cat([values[1][idx + k1]]*int(fill[idx+k1]),0) for k1 in range(0,fill_mask.sum())]
                   values[1] = tuple(values[1])

                   values[2] = list(values[2])
                   values[2][idx:] = [torch.cat([values[2][idx + k1]]*int(fill[idx+k1]),0) for k1 in range(0,fill_mask.sum())]
                   values[2] = tuple(values[2])

                   values[7] = list(values[7])
                   values[7][idx:] = [torch.zeros(values[2][idx+k1].shape) for k1 in range(0,fill_mask.sum())]
                   values[7] = tuple(values[7])

                   sequence_mask = [values[6][idx + k1] < (value[idx+k1].shape[0]) - (value[idx+k1].shape[0])/int(fill[idx+k1]) for k1 in range(0,fill_mask.sum())]
                   values[6] = list(values[6])
                   values[6][idx:] = [torch.cat([values[6][idx + k1],torch.Tensor(np.arange(values[6][idx + k1][-1] + 1,values[6][idx + k1][-1] + 1 + (value[idx+k1].shape[0]) - (value[idx+k1].shape[0])/int(fill[idx+k1]), 1))],0) for k1 in range(0,fill_mask.sum())]
                   values[6] = tuple(values[6])    

                # padd the tensors when the self-filling is not sufficient to complet the required size
                dif = torch.Tensor([value[0].shape[0] - value[i].shape[0]  for i in range(0,len(value))])
                
                dif_mask = dif > 0

                if dif_mask.any():

                   idx = np.where(dif_mask)[0][0]
                   # pad if sequences have different lengths
                   value = list(value)
                   value[idx:] = [torch.cat([value[idx + k1], value[idx + k1][:dif[idx+k1].type(torch.int32)]], 0) for k1 in range(0,dif_mask.sum())]
                   value = tuple(value)
                   # pad the other tensors according to the difference on the number of frames
                   # position of the label corresponding to a frame from an individual clip (vid_pos)
                   sequence_mask = [values[5][idx + k1] < dif[idx + k1] for k1 in range(0,dif_mask.sum())]
                   values[5] = list(values[5])
                   values[5][idx:] = [torch.cat([values[5][idx + k1],values[5][idx + k1][sequence_mask[k1]] + value[idx+ k1].shape[0]],0 ) for k1 in range(0,dif_mask.sum())]
                   values[5] = tuple(values[5])

                   # box corresponding to each frame from each clip
                   values[1] = list(values[1])

                   values[1][idx:] = [torch.cat([values[1][idx + k1],values[1][idx + k1][sequence_mask[k1]]],0 ) for k1 in range(0,dif_mask.sum())]
                   values[1] = tuple(values[1])

                   # cls corresponding to each frame from each clip
                   values[2] = list(values[2])
                   values[2][idx:] = [torch.cat([values[2][idx + k1],values[2][idx + k1][sequence_mask[k1]]],0 ) for k1 in range(0,dif_mask.sum())]
                   values[2] = tuple(values[2])
                   
                   
                   # filling the corresponding batches 
                   values[7] = list(values[7])
                   values[7][idx:] = [torch.zeros(values[2][idx+k1].shape) for k1 in range(0,dif_mask.sum())]
                   values[7] = tuple(values[7])

                   # position of the frame in the video (clip_pos)
                   sequence_mask = [values[6][idx + k1] < dif[idx + k1] for k1 in range(0,dif_mask.sum())]
                   values[6] = list(values[6])
                   values[6][idx:] = [torch.cat([values[6][idx + k1],torch.Tensor(np.arange(values[6][idx + k1][-1] + 1, values[6][idx + k1][-1] + 1 + dif[idx + k1], 1))],0) for k1 in range(0,dif_mask.sum())]
                   values[6] = tuple(values[6])    



                   


                value = torch.stack(value, 0)
                
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'clip_pos', 'vid_pos']:
                value = torch.cat(value, 0)
            new_batch[k] = value

        
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

    @staticmethod
    def collate_fn(batch):
        #start_collate = time.time()
        new_batch = {}
        keys = batch[0].keys()
        
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            
            value = values[i]
            if k == 'img':

                value = torch.stack(value, 0)
                
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'clip_pos', 'vid_pos']:
                value = torch.cat(value, 0)
            new_batch[k] = value

        
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        #print("collate time:", time.time()-start_collate)
        return new_batch



