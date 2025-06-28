# Ultralytics YOLO 🚀, GPL-3.0 license
import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks,
                                    colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.ops import Profile
import torch.nn as nn
from ultralytics.yolo.data.utils import check_det_dataset
from EventVideoDataloader import build_video_dataloader, build_video_val_standalone_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images, plot_event_images
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
import argparse 
import yaml
from ultralytics.nn.autobackend import AutoBackendMemory
import math


class EventVideoDetectionValidator(BaseValidator):
    #clip_length = 21, clip_stride = 21
    def __init__(self, video_config = None, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)

        self.args = args or get_cfg(DEFAULT_CFG)
        self.video_config = video_config
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.imgsz = self.args.imgsz
        self.dtype = torch.cuda.HalfTensor if self.args.half else torch.cuda.FloatTensor
        if self.args.plots:
          if not os.path.exists(os.path.join(self.save_dir, "preds")):
            os.mkdir(os.path.join(self.save_dir, "labels"))
            os.mkdir(os.path.join(self.save_dir, "preds"))
        if not hasattr(args,"show_sequences"):
           self.args.show_sequences = -1
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.dtype = torch.cuda.HalfTensor if self.args.half else torch.cuda.FloatTensor
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots = trainer.epoch == trainer.epochs - 1  # always plot final epoch
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != 'cpu'
            model = AutoBackendMemory(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

            #imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    self.logger.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch,  rank = -1, load = "sequential", speed = self.args.speed, zero_hidden = self.args.zero_hidden)
            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.args.channels, self.imgsz, self.imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
         
          batch["img"] = batch["img"].float()
          hidden_states = {"0": None, "1": None, "2": None, "3": None}
          
          for T in range(batch["img"].shape[1]):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            sequence_mask = batch["vid_pos"] == T
            # preprocess
            #batch, classes, bboxes, batch_idx
            with dt[0]:
                batch_ = self.preprocess(batch['img'][:,T,:,:,:], batch['cls'][sequence_mask], batch['bboxes'][sequence_mask], batch['batch_idx'][sequence_mask]).type(self.dtype)
                
            # inference
            with dt[1]:
                
                preds, hidden_states = model(batch_, hidden_states)

                #print(len(preds))
            # loss
            with dt[2]:
                if self.training:
                    self.loss += trainer.criterion(preds, batch,sequence_mask, None)[1]

            # postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch_, batch,sequence_mask, T)
            if self.args.plots and batch_i < self.args.show_sequences:
                self.plot_val_samples(batch_, batch,batch_i,T,sequence_mask)
                self.plot_predictions(batch_, batch,preds, batch_i,T)

            self.run_callbacks('on_val_batch_end') 

        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = tuple(x.t / len(self.dataloader.dataset) * 1E3 for x in dt)  # speeds per image
        self.finalize_metrics()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            self.logger.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                             self.speed)
           
            return stats


    def preprocess(self, batch, classes, bboxes, batch_idx):
        batch = batch.to(self.device, non_blocking=True)
        batch = (batch).float()

        new_scale = [batch.shape[2] + (math.ceil(batch.shape[2]/32)*32 -  batch.shape[2]), batch.shape[3] + (math.ceil(batch.shape[3]/32)*32 - batch.shape[3])]
       
        batch = nn.functional.interpolate(batch,scale_factor = (new_scale[0] / batch.shape[2], new_scale[1] / batch.shape[3]), mode = 'bilinear')
        classes = classes.to(self.device)
        bboxes = bboxes.to(self.device)
        batch_idx = batch_idx.to(self.device)

        nb = (batch.shape[0])
        self.lb = [torch.cat([classes, bboxes], dim=-1)[batch_idx == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

    
        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))

        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0

        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        
        
        preds = ops.non_max_suppression(preds,
                                        self.args.conf, #self.args.conf
                                        self.args.iou,  #self.args.iou
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=False,
                                        max_det=self.args.max_det)
        return preds

    def update_metrics(self, preds, batch_,batch, sequence_mask, T):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'][sequence_mask] == si
            cls = batch['cls'][sequence_mask][idx]
            cls = cls.view(cls.shape[0],1).to(self.device)
            bbox = batch['bboxes'][sequence_mask][idx].to(self.device)
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions

            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:

                if nl:


                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            #if self.args.single_cls:
            #    pred[:, 5] = 0
            predn = pred.clone()
            #ops.scale_boxes(batch['img'][:,T,:,:,:][si].shape[1:], predn[:, :4], shape)#,ratio_pad=batch['ratio_pad'][si])  # native-space pred
            ops.scale_boxes(batch_[si].shape[1:], predn[:, :4], batch_[si].shape[1:])#,ratio_pad=batch['ratio_pad'][si])  # native-space pred
            # Evaluate
            if nl:
                height, width = batch_.shape[2:]
                tbox = ops.xywh2xyxy(bbox).to(self.device) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                #ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape) #ratio_pad=batch['ratio_pad'][si])  # native-space labels
                ops.scale_boxes(batch_[si].shape[1:], predn[:, :4], batch_[si].shape[1:])#ratio_pad=batch['ratio_pad'][si])  # native-space labels 
                #labelsn = torch.cat((cls.to(self.device).view(cls.shape[0],1), tbox), 1)  # native-space labels
                labelsn = torch.cat((cls.to(self.device), tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)

            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls))

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = dict(zip(self.metrics.speed.keys(), self.speed))

    def get_stats(self):
        
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        #stats = [torch.stack(x, 0).cpu().numpy() for x in zip(*self.stats)]
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)

        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):

        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        self.logger.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            self.logger.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                self.logger.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:

            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def get_dataloader(self, dataset_path, batch_size,  mode='val', rank=0, load = 'sequential', speed = False, zero_hidden = False):

        self.video_config = {"clip_length": self.args.clip_length, "clip_stride": self.args.clip_length, "channels": self.args.channels}
        return build_video_val_standalone_dataloader(self.args, self.video_config, batch_size,dataset_path,rank=rank, mode = load, speed = speed, zero_hidden = zero_hidden)[0]

    def plot_val_samples(self, batch_, batch, ni,si, seq_mask):

        plot_event_images(batch_,
                    batch['batch_idx'][seq_mask],
                    batch['cls'][seq_mask].squeeze(-1),
                    batch['bboxes'][seq_mask],
                    paths=None,
                    fname=self.save_dir / f'labels/val_batch{ni}_seq{si}_labels.jpg',
                    names=self.names)

    def plot_predictions(self, batch_, batch, preds, ni,si):

        plot_event_images(batch_,
                    *output_to_target(preds, max_det=15),
                    paths=None,
                    fname=self.save_dir / f'preds/val_batch{ni}_seq{si}_pred.jpg',
                    names=self.names)  # pred

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            self.logger.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocococo import COCO  # noqa
                from pycocococoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                self.logger.warning(f'pycocotools unable to run: {e}')
        return stats



###### Code for Standalone Usage ##############
def val(cfg=DEFAULT_CFG,use_python=False):
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
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--channels', type=int, default=1, help='number of channels')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', action = 'store_true')
    parser.add_argument('--save_txt', action = 'store_true')
    parser.add_argument('--save_conf', action = 'store_true')
    parser.add_argument('--save_crop', action = 'store_true')
    parser.add_argument('--hide_labels', action = 'store_true')
    parser.add_argument('--half', action='store_true', help='use FP16 format')
    parser.add_argument('--hide_conf', action = 'store_true')
    parser.add_argument('--dnn', action = 'store_true')
    parser.add_argument('--save_hybrid', action = 'store_true')
    parser.add_argument('--conf', default = None, type=float)
    parser.add_argument('--iou', default = 0.7, type=float)
    parser.add_argument('--max_det', default = 300, type=int)
    parser.add_argument('--speed', action='store_true')
    parser.add_argument('--zero_hidden', action='store_true')
    parser.add_argument('--split', default = 'val', type=str)
    parser.add_argument('--plots', action='store_false')
    #parser.add_argument('--single_cls', action='store_true')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--clip_length', default = 1, type=int)
    parser.add_argument('--clip_stride', default = 1, type=int)
    parser.add_argument('--show_sequences', default = 3, type=int)
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


 args = parse_opt()
 args.project = args.project / args.split

 validator = EventVideoDetectionValidator(args=args)
 validator(model=args.model)


if __name__ == '__main__':
    val()
