import sys 
import torch
import numpy as np 
import os
import random
import copy
import torch
import subprocess
import torch.nn as nn
import time
from collections import defaultdict
from copy import deepcopy
from copy import copy
import wandb
import torch.nn as nn
from datetime import datetime

from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from EventVideoDataloader import build_video_dataloader, build_video_val_standalone_dataloader
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.utils import  PIN_MEMORY, RANK
from ultralytics.nn.tasks import DetectionModel2
from ultralytics.yolo import v8

from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel
import numpy as np
import val
from tqdm import tqdm
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks,
                                    colorstr, emojis, yaml_save)
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
import argparse 
import yaml


from pathlib import Path
import math
######################### ADDING THE ARG PARSE ##############################################
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov8n.pt', help='initial weights path')
    parser.add_argument('--model', type=str, default=ROOT / 'yolov8n.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--nbs', type=int, help='nominal batch size', default = 16)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)',nargs='+')
    parser.add_argument('--seed',type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--save_period',type=int, default=-1, help='save checkpoint every x epochs, disabled if -1')
    parser.add_argument('--save',action='store_false', help='save train checkpoints and predict results')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--cos_lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--half', action='store_true', help='use FP16 format')
    parser.add_argument('--plots', action='store_false', help='plot results')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    # Hyperparameters 
    parser.add_argument('--hyp', type=str, default= ROOT / 'ultralytics/yolo/cfg/default.yaml', help='hyperparameters path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # Video Hyperparameters
    parser.add_argument('--clip_length', type=int, default=11)
    parser.add_argument('--clip_stride', type=int, default=11)
    parser.add_argument('--channels',  type=int, default=1)  
    parser.add_argument('--val_epoch',  type=int, default=1)  
    # Augmentation Hyperparameters
    parser.add_argument('--flip', type=float, default=0.0)
    parser.add_argument('--invert',  type=float, default=0.0)  
    parser.add_argument('--suppress',  type=float, default=0.0)  
    parser.add_argument('--positive',  type=float, default=0.0)  
    parser.add_argument('--zoom_out',  type=float, default=0.0)  
    parser.add_argument('--max_zoom_out_factor',  type=float, default=2.0)  
    parser.add_argument('--min_zoom_out_factor',  type=float, default=1.0)  



    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

args = parse_opt()
# Open hyparparameter files
overrides = yaml.safe_load(Path(args.hyp).read_text())
# append arg_parse items to the overrides dictionary

cfg = "default.yaml"
overrides["save"] = args.save

overrides["save_period"] = args.save_period
overrides["model"] = args.model
overrides["seed"] = args.seed
overrides["data"] = args.data
overrides["epochs"] = args.epochs
overrides["batch"] = args.batch
overrides["imgsz"] = args.imgsz
overrides["rect"] = args.rect
overrides["resume"] = args.resume
overrides["cache"] = args.cache
overrides["device"] = args.device
overrides["workers"] = args.workers
overrides["project"] = args.project
overrides["name"] = args.name
overrides["cos_lr"] = args.cos_lr
overrides["half"] = args.half
overrides["plots"] = args.plots
overrides["pretrained"] = args.pretrained
overrides["nbs"] = args.nbs
overrides["optimizer"] = args.optimizer


overrides["clip_length"] = args.clip_length
overrides["clip_stride"] = args.clip_length
overrides["channels"] = args.channels

overrides["val_epoch"] = args.val_epoch

overrides["flip"] = args.flip
overrides["invert"] = args.invert
overrides["suppress"] = args.suppress
overrides["positive"] = args.positive
overrides["zoom_out"] = args.zoom_out
overrides["max_zoom_out_factor"] = args.max_zoom_out_factor
overrides["min_zoom_out_factor"] = args.min_zoom_out_factor


# BaseTrainer python usage
class EventVideoYOLOv8DetectionTrainer(BaseTrainer):
    
    def __init__(self,cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.console = LOGGER
        self.validator = None
        self.model = None
        self.metrics = None  
        
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        
 
        self.video_config = {"clip_length": args.clip_length, "clip_stride": args.clip_stride, "channels": args.channels}
        self.aug_params = {"flip": args.flip, "invert": args.invert, "suppress": args.suppress, "positive": args.positive, "zoom_out": args.zoom_out, "max_zoom_out_factor": args.max_zoom_out_factor, 
        "min_zoom_out_factor": args.min_zoom_out_factor}
        
        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in {-1, 0} else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        self.amp = self.device.type != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        if RANK in {0, -1}:
            callbacks.add_integration_callbacks(self)
            wandb.init(project =  self.args.project, name = self.args.name, config=
        overrides)
        


    
    def get_dataloader(self, dataset_path, batch_size, aug_param, mode, rank=0, load = "batched"):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)


        if mode == "train":
         return build_video_dataloader(self.args, self.video_config, batch_size,dataset_path, aug_param = self.aug_params,rank=rank, mode = mode, load = load)[0]
        else: 

           return build_video_val_standalone_dataloader(self.args, self.video_config, batch_size,dataset_path,rank, mode = load)[0]

    def preprocess_batch(self, batch):
        #batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        #batch = batch.to(self.device, non_blocking=True).float() / batch.max()
        #batch = (batch*127.5 + 127.5).to(self.device, non_blocking=True).float() / 255   
        batch = (batch).to(self.device, non_blocking=True).float()  

        #print("before interp", batch.shape)
        new_scale = [batch.shape[2] + (math.ceil(batch.shape[2]/32)*32 -  batch.shape[2]), batch.shape[3] + (math.ceil(batch.shape[3]/32)*32 -  batch.shape[3])]

        batch = nn.functional.interpolate(batch,scale_factor = (new_scale[0] / batch.shape[2], new_scale[1] / batch.shape[3]), mode = 'bilinear')

        return batch

    def set_model_attributes(self):
        # TO DO: IT SHOULD BE BETTER TO ADD MODEL CHANNELS HERE
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel2(cfg, imgsz = self.args.imgsz,ch=self.video_config["channels"], nc=self.data['nc'], verbose=True)
        if weights:
            model.load(weights)

        return model
    
    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'

        return val.EventVideoDetectionValidator(self.video_config, self.test_loader,save_dir=self.save_dir,logger=self.console,args=copy(self.args))

    def criterion(self, preds, batch, sequence_mask, cur_loss):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = LossVideo(de_parallel(self.model))
        return self.compute_loss(preds, batch, sequence_mask, cur_loss)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        # TO DO: CHANGE IT TO A FUNCTION THAT IS SPECIALIZED ON VOXELS
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png


    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process.
        """
        # model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        if world_size > 1:
            #self.model = DDP(self.model, device_ids=[rank], find_unused_parameters = True)
            self.model = DDP(self.model, device_ids=[rank], broadcast_buffers=False)
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        #self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        # Batch size
 
        # Optimizer
        self.accumulate = max(round((self.args.nbs / self.video_config["clip_length"] )/ self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / (self.args.nbs / self.video_config["clip_length"] )# scale weight_decay
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # dataloaders
        self.batch_size_ = self.batch_size // world_size if world_size > 1 else self.batch_size
        #get_dataloader(self, dataset_path, batch_size, img_x, img_y, mode='train', rank=0)

        self.train_loader = self.get_dataloader(self.trainset, batch_size=self.batch_size_, aug_param = self.aug_params, rank=rank, mode='train')

        if rank in {0, -1}:
               
            self.test_loader = self.get_dataloader(self.testset, batch_size=self.batch_size_ * 2, aug_param = self.aug_params,mode='val',rank=-1, load = "batched")
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

    def get_test_dataset(self, data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data.get('test')


    def final_eval(self):
        del self.testset, self.trainset
        self.testset = self.get_test_dataset(self.data)
        #self, dataset_path, batch_size, img_x, img_y, aug_param, mode, rank=0, load = "batched", mixed_load = False
        self.test_loader = self.get_dataloader(self.testset, batch_size=self.batch_size_ * 2,aug_param = self.aug_params,mode='val',rank=-1, load = "sequential")
        self.validator = self.get_validator()

        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    self.console.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')
                    wandb.log(self.metrics)


    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)
    
        self._setup_train(rank, world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        self.log(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                 f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                 f"Logging results to {colorstr('bold', self.save_dir)}\n"
                 f'Starting training for {self.epochs} epochs...')

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            

            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            
            if rank in {-1, 0}:
                self.console.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None

            for i, batch in pbar:        
             hidden_states = {"0": None, "1": None, "2": None, "3": None}
             #self.loss = torch.zeros([], device=self.device)
             self.optimizer.zero_grad()
             for T in range(self.video_config["clip_length"]):
                sequence_mask = batch['vid_pos'] == T
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, (self.args.nbs / self.video_config["clip_length"])/ self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])


                with torch.cuda.amp.autocast(self.amp):

                    batch_ = self.preprocess_batch(batch['img'][:,T,:,:,:])

                    preds, hidden_states = self.model(batch_, hidden_states)

                    if T == 0:
                     self.loss, self.loss_items = self.criterion(preds, batch, sequence_mask, None)
                    else: 

                     self.loss, self.loss_items = self.criterion(preds, batch, sequence_mask, self.loss)

                    
                    if rank != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items


             self.scaler.scale(self.loss).backward()

             # Optimizer Step only at the end of sequence
             # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
             if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

             # Log
             mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
             loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
             losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
             if rank in {-1, 0}:
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    #if self.args.plots and ni in self.plot_idx:
                    #    self.plot_training_samples(batch, ni)

             self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if rank in {-1, 0}:

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if (self.args.val and (epoch - 1) % self.args.val_epoch == 0  and epoch != 0):
                    self.metrics, self.fitness = self.validate()         



                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if rank in {-1, 0}:
            # Do final val with best.pt
            self.log(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                     f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.log(f"Results saved to {colorstr('bold', self.save_dir)}")
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')



# Criterion class for computing training losses
class LossVideo:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        roll_out_thr = h.min_memory if h.min_memory > 1 else 64 if h.min_memory else 0  # 64 is default

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):

        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))

        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, sequence_mask, cur_loss):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds


        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets

        targets = torch.cat((batch['batch_idx'][sequence_mask].view(-1, 1), batch['cls'][sequence_mask].view(-1, 1), batch['bboxes'][sequence_mask]), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if cur_loss:
           return loss.sum() * batch_size + cur_loss, loss.detach()  # loss(box, cls, dfl)
        else: 
           return loss.sum() * batch_size, loss.detach()

trainer = EventVideoYOLOv8DetectionTrainer(overrides=overrides)
trainer.train()
