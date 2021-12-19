import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import tensorboard

import numpy as np
import random
import os
import sys
from tqdm import tqdm

sys.path.append('.')

from configs import parse_args
from utils.criterion import *
from utils.datasets import CocoDetection
from models.yolo import Model
from utils.general import non_max_suppression, make_box

if __name__ == '__main__':
    torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    writer = SummaryWriter('./log')
    args = parse_args()
    # print(args)
    
    model = Model(cfg='models/yolov5m.yaml', ch=3, nc=1)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join('./checkpoints', 'ck.pth')))

    valset = CocoDetection(args.dataset, 'val', args.size)

    val_iter = DataLoader(valset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=16,
                        collate_fn=valset.collate_fn,
                        pin_memory=True)
    
    print("Done Pre.")
    pbar = tqdm(total = valset.__len__())
    
    model.eval()
    for i, (img, targets) in enumerate(val_iter):
        img, targets = img.cuda(), targets.cuda()
        p, _ = model(img)
        pred = non_max_suppression(p)[0]
        try:
            pred_box_tensor, score = pred[:,:4], pred[:,4]
            gt_box_tensor = make_box(targets[:,2:])
            img = img.squeeze(0)
            score_str = [str(x)[:5] for x in score.tolist()]
            writer.add_image_with_boxes('gt', img, gt_box_tensor, global_step=i)
            writer.add_image_with_boxes('pred', img, pred_box_tensor, global_step=i, labels=score_str)
        except:
            pass
        pbar.update(1)

    writer.close()
       