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
from models.net import Net

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
    
    train_iter = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    for img, targets in train_iter:
        continue
    