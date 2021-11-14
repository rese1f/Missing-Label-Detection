import torch
import torch.nn as nn
# This document describes the loss function
class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
    
    def forward(self, pred, target):
        return