import numpy as np
import os 
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import matplotlib 
import matplotlib.pyplot as plt 



def create_destination_folder(destination,formatInfo,name, size_x, size_y):
    destFolder = os.path.join(destination,name + '_' + formatInfo["method"] + '_' + str(size_x) + '_' + str(size_y) + '_' + str(formatInfo["timeWindow"]) +  "ms_bins_" + str(formatInfo["tbin"]))
   
    if not(os.path.exists(destFolder)):
           os.makedirs(destFolder, exist_ok = True)
           imageFolder = os.path.join(destFolder,'images')
           os.makedirs(imageFolder, exist_ok = True)
           os.makedirs(os.path.join(imageFolder,'train'), exist_ok = True)
           os.makedirs(os.path.join(imageFolder,'test'), exist_ok = True)
           os.makedirs(os.path.join(imageFolder,'val'), exist_ok = True)
           labelFolder = os.path.join(destFolder,'labels')
           os.makedirs(labelFolder, exist_ok = True)
           os.makedirs(os.path.join(labelFolder,'train'), exist_ok = True)
           os.makedirs(os.path.join(labelFolder,'test'), exist_ok = True)
           os.makedirs(os.path.join(labelFolder,'val'), exist_ok = True)
    return destFolder

def filter_boxes(boxes, skip_ts= 0, min_box_diag=0, min_box_side=0, dataset = "GEN1"):
    """Filters boxes according to the paper rule.  From Prophesee
    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0
    Args:
        boxes (np.ndarray): structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence'] 
        (example BBOX_DTYPE is provided in src/box_loading.py)
    Returns:
        boxes: filtered boxes
    """
    ts = boxes['t'] 
    width = boxes['w']
    height = boxes['h']
    classes = boxes['class_id']
    diag_square = width**2+height**2
    if dataset == "1MP_3classes":
     mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)*(((classes == 0) | (classes == 1) | (classes == 2)))
    elif dataset == "1MP_7classes":
     mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)*(((classes == 0) | (classes == 1) | (classes == 2) | (classes == 3) | (classes == 4) | (classes == 5) | (classes == 6)))
    else:
     mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)*(((classes == 0) | (classes == 1)))
    return boxes[mask]


def clip_boxes(boxes, width, height):
        
        boxes['w'] = np.clip(boxes['w'],0,width)        
        boxes['h'] = np.clip(boxes['h'],0,height) 
        
        mask_x = boxes['x'] < 0
        boxes['w'][mask_x] = np.clip(boxes['w'][mask_x]  + boxes['x'][mask_x],1,width) 
        boxes['x'][mask_x] = 0 
        
        mask_y = boxes['y'] < 0
        boxes['h'][mask_y] = np.clip(boxes['h'][mask_y]  + boxes['y'][mask_y],1,height) 
        boxes['y'][mask_y] = 0 
        
        mask_x = (boxes['x'] + boxes['w']) > width
        mask_y = (boxes['y'] + boxes['h']) > height


        boxes['w'][mask_x] =  width - (boxes['x'][mask_x])
        boxes['h'][mask_y] =  height - (boxes['y'][mask_y]) 

        return boxes 

def to_bbox_yolo_format(box_events, width, height):
        box_events['x'] = (box_events['x'] + box_events['w']/2)/width
        box_events['y'] = (box_events['y'] + box_events['h']/2)/height
        box_events['w'] = box_events['w']/width
        box_events['h'] = box_events['h']/height
        box_events['class_id'] = box_events['class_id'] 
        return(box_events)


def create_labels(box_events):
    labels = np.zeros((len(box_events['x']),5))
    labels[:,0] = box_events['class_id']
    labels[:,1] = box_events['x']
    labels[:,2] = box_events['y']
    labels[:,3] = box_events['w']
    labels[:,4] = box_events['h']
    return labels

def save_hist(destFolder, index, category, fileList, box_events, frame): 

           plt.imsave(destFolder + '/images/'+ category+'/figurinha_list' + fileList + '_' + str(index) +'.jpeg',frame) 
           with open(destFolder + '/labels/'+category+'/figurinha_list'+ fileList + '_' + str(index) +'.txt', 'w') as f:
             for idx in range(0,len(box_events['x'])):
                f.write(str(box_events['class_id'][idx]))
                f.write(' ')
                f.write(str(box_events['x'][idx]))
                f.write(' ')
                f.write(str(box_events['y'][idx]))
                f.write(' ')
                f.write(str(box_events['w'][idx]))
                f.write(' ')
                f.write(str(box_events['h'][idx]))
                f.write(' ')
                f.write('\n')
                
           f.close() 



def save_compressed_clip(destFolder, index, category, fileList, frame, labels, compress = True): 
   f = os.path.join(destFolder + '/images/'+ category+'/sequence_' + fileList + '_subseq_' + str(index).zfill(7) +'.h5')
   if compress:
    hf = h5py.File(f,'w')
    hf.create_dataset('1mp', data = frame,**hdf5plugin.Blosc(cname='zstd'))
    hf.close()
   else:
    hf = h5py.File(f,'w')
    hf.create_dataset('1mp', data = frame)
    hf.close()
   
   
   np.save(os.path.join(destFolder + '/labels/'+ category+'/sequence_' + fileList + '_subseq_' + str(index).zfill(7) +'.npy'),np.array(labels, dtype = object))


def save_only_compressed_clip(destFolder, index, category, fileList, frame, compress = True): 
   f = os.path.join(destFolder + '/images/'+ category+'/sequence_' + fileList + '_subseq_' + str(index).zfill(6) +'.h5')
   if compress:
    hf = h5py.File(f,'w')
    hf.create_dataset('1mp', data = frame,**hdf5plugin.Blosc(cname='zstd'))
    hf.close()
   else:
    hf = h5py.File(f,'w')
    hf.create_dataset('1mp', data = frame)
    hf.close()



def save_compressed_clip_label(destFolder, k, category, fileList, labels, compress = True): 

   np.save(os.path.join(destFolder + '/labels/'+ category+'/sequence_' + fileList + '_subseq_' + str(k).zfill(7) +'.npy'),np.array(labels, dtype = object))




