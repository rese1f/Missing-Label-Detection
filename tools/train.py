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
from utils.general import non_max_suppression, make_box, calculate_ap

if __name__ == '__main__':
    torch.set_printoptions(precision=None, threshold=4096, edgeitems=None, linewidth=None, profile=None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    args = parse_args()
    print(args)
    writer = SummaryWriter(os.path.join('./log', args.name))
    
    model = Model(cfg='models/yolov5m.yaml', ch=3, nc=1)
    model = nn.DataParallel(model).cuda()
    if args.checkpoint:
        model.load_state_dict(torch.load(os.path.join('./checkpoints', args.checkpoint)))

    trainset = CocoDetection(args.dataset, 'val', args.size)
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
                        collate_fn=valset.collate_fn,
                        pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # lf = lambda x: ((1 + np.cos(x * np.pi / args.epoch)) / 2) * (1 - 0.12) + 0.12  # cosine
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    print("Done Pre.")
    pbar = tqdm(total = args.epoch)
    
    for epoch in range(args.epoch):
        model.train()
        loss_list, box_loss_list, obj_loss_list, lr_list = list(), list(), list(), list()
        # writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        for img, targets in train_iter:
            img, targets = img.cuda(), targets.cuda()
            p = model(img)
            compute_loss = ComputeLoss(model)
            loss, losses = compute_loss(p, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())
            box_loss_list.append(losses[0].detach().cpu().numpy())
            obj_loss_list.append(losses[1].detach().cpu().numpy())
        # scheduler.step()
        loss, box_loss, obj_loss = np.mean(loss_list), np.mean(box_loss_list), np.mean(obj_loss_list)
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('box_loss', box_loss, epoch)
        writer.add_scalar('obj_loss', obj_loss, epoch)
        
        if args.val and (epoch % 10 == 0):
            model.eval()
            ap50_list = list()
            for i, (img, targets) in enumerate(val_iter):
                img, targets = img.cuda(), targets.cuda()
                p, _ = model(img)
                pred = non_max_suppression(p)
                pred_box_tensor, score = pred[:,:4], pred[:,4]
                gt_box_tensor = make_box(targets[:,2:])
                ap50 = calculate_ap(pred_box_tensor, score, gt_box_tensor,iou_thresh = 0.5)
                ap50_list.append(ap50)
            print("ap50:", np.mean(ap50_list))
            writer.add_scalar('ap50', np.mean(ap50_list), epoch)
        pbar.update(1)
        
    writer.close()
    torch.save(model.state_dict(), os.path.join('checkpoints', args.name+'.pth'))