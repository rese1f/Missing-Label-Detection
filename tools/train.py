import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import random
import os
import sys
sys.path.append('.')

from configs import parse_args
from utils.criterion import *
from utils.datasets import CocoDetection
from models.yolo import Model

if __name__ == '__main__':
    torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    writer = SummaryWriter('./log')
    args = parse_args()
    print(args)

    trainset = CocoDetection(args.dataset, 'train', args.size)
    valset = CocoDetection(args.dataset, 'val', args.size)
    
    train_iter = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=trainset.collate_fn,
                            pin_memory=True)
    # for img, targets in train_iter:
        # print('img shape:',np.shape(img))
        # print('print target',targets)
        # continue
    val_iter = DataLoader(valset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=trainset.collate_fn,
                        pin_memory=True)
    # for img, targets in val_iter:
        # print(targets)
        # continue
    
    # # Print statistic info of img shape before and after interpolation
    # print('trainset.shape_before_dic',trainset.shape_before_dic)
    # print('trainset.shape_after_dic',trainset.shape_after_dic)
    # print('valset.shape_before_dic',valset.shape_before_dic)
    # print('valset.shape_after_dic',valset.shape_after_dic)
    
    # # print the number of img with no label
    # print('train:',trainset.empty_label_number) 


    model = Model(cfg='models/yolov5m.yaml', ch=3, nc=1)