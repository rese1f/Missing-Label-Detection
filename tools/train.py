import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import random
import os
import sys
import torch.nn as nn
sys.path.append('.')

from configs import parse_args
from utils.criterion import *
from utils.datasets import CocoDetection
from models.yolo import Model


if __name__ == '__main__':
    torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    writer = SummaryWriter('./log')
    args = parse_args()
    print(args)
    
    model = Model(cfg='models/yolov5m.yaml', ch=3, nc=1)
    model = nn.DataParallel(model).cuda()

    trainset = CocoDetection(args.dataset, 'train', args.size)
    valset = CocoDetection(args.dataset, 'val', args.size)
    train_iter = DataLoader(trainset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=16,
                            collate_fn=trainset.collate_fn,
                            pin_memory=True)
    val_iter = DataLoader(valset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=16,
                        collate_fn=trainset.collate_fn,
                        pin_memory=True)

    for epoch in range(args.epoch):
        model.train()
        for img, targets in train_iter:
            img, targets = img.cuda(), targets.cuda()
            p = model(img)
            compute_loss = ComputeLoss(model)
            import pdb
            pdb.set_trace()
            break
        break