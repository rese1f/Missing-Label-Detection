from random import sample
import torch
import torch
from torch import tensor
from torch.utils.data import Dataset

class COCO(Dataset):
    def __init__(self) -> None:
        super(COCO, self).__init__()

    def __getitem__(self, index) -> tuple:
        return (0,0)
    
    def __len__(self) -> int:
        return 0