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
    print(args)
    
    model = Model(cfg='models/yolov5m.yaml', ch=3, nc=1)
    model = nn.DataParallel(model).cuda()
    # if args.checkpoint:
    #     model.load_state_dict(torch.load(os.path.join('./checkpoints', args.checkpoint)))

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
                        collate_fn=valset.collate_fn,
                        pin_memory=True)

    SGD = optim.SGD(model.parameters(), lr=args.lr)
    
    print("Done Pre.")
    pbar = tqdm(total = args.epoch)
    
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
        
        model.eval()
        if args.val and (epoch % 20 == 10):
            for i, (img, targets) in enumerate(val_iter):
                img, targets = img.cuda(), targets.cuda()
                p, _ = model(img)
                pred = non_max_suppression(p)
                if (i == 0):
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
    torch.save(model.state_dict(), 'checkpoints/baseline.pth')
       