import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import tensorboard
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

    SGD = optim.SGD(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epoch):
        model.train()
        loss_list, box_loss_list, obj_loss_list = list(), list(), list()
        for img, targets in train_iter:
            img, targets = img.cuda(), targets.cuda()
            p = model(img)
            compute_loss = ComputeLoss(model)
            loss, losses = compute_loss(p, targets)
            SGD.zero_grad()
            loss.backward()
            SGD.step()
            loss_list.append(loss.detach().cpu().numpy())
            box_loss_list.append(losses[0].detach().cpu().numpy())
            obj_loss_list.append(losses[1].detach().cpu().numpy())
        loss, box_loss, obj_loss = np.mean(loss_list), np.mean(box_loss_list), np.mean(obj_loss_list)
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('box_loss', box_loss, epoch)
        writer.add_scalar('obj_loss', obj_loss, epoch)
    writer.close()