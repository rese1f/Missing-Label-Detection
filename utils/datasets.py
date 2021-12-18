import os
import os.path
from PIL import Image
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F 
from pycocotools.coco import COCO 
import matplotlib.pyplot as plt


class CocoDetection(data.Dataset):

    def __init__(self, root, stage, img_size):
        """
        Discription: Initialize the class CocoDetection.

        Args:
            root (string): Root directory where images are downloaded to.
            stage (string): Path to json annotation file.
            img_size (int): Target image size.
            
        Return:
            null
        """
        super(CocoDetection,self).__init__()
        print('=====================')
        print('Loading {} dataset...'.format(stage))
        self.root = os.path.join(root, stage) 
        self.coco = COCO(os.path.join(root, 'annotation', stage + '.json'))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size
        self.batch_count = 0
        self.empty_label_number = 0 # the number of imgs with no annotation
        self.shape_before_dic = {}
        self.shape_after_dic = {}


    def __getitem__(self, index):


        coco = self.coco
        img_id = self.ids[index]
        #==============
        # image
        # =============
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root,path)).convert('RGB')
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        c, h, w = img.shape
    

        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding（左，右，上，下）
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=0)
        _, padded_h, padded_w = img.shape

        # Resize
        # print('before',np.shape(img))
        if(np.shape(img) in self.shape_before_dic):
            self.shape_before_dic[np.shape(img)] += 1
        else:
            self.shape_before_dic[np.shape(img)] = 1
            
        img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)
        
        # print('after',np.shape(img))
        if(np.shape(img) in self.shape_after_dic):
            self.shape_after_dic[np.shape(img)] += 1
        else:
            self.shape_after_dic[np.shape(img)] = 1
        #==============
        # labels
        # =============
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)

        bboxes = []
        for i in range(len(anns)):
            bbox = [anns[i]['category_id']-1]
            bbox.extend(anns[i]['bbox']) # (x,y,w,h) x和y表示bbox左上角的坐标，w和h表示bbox的宽度和高度
            bboxes.append(bbox)
        
        bboxes = torch.from_numpy(np.array(bboxes))

        # TODO To handle no target imgae. For now, add a box cover the whole img
        if(np.shape(bboxes)==torch.Size([0])):
            return img, None
            # add a box target cover the whole img
            bboxes = torch.tensor([[  1.0000, 0.0000, 0.0000,  w,  h]])
            self.empty_label_number += 1
            
        # Extract coordinates for unpadded + unscaled image（这好像计算出来的是bbox左上和右下两点的坐标）
        x1 = (bboxes[:, 1])
        y1 = (bboxes[:, 2])
        x2 = (bboxes[:, 1] + bboxes[:, 3])
        y2 = (bboxes[:, 2] + bboxes[:, 4])
        # Adjust for added padding（调整padding后两点的坐标）
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)（重新归一化，（x,y）表示中心点坐标，（w,h）表示bbox的宽和高）
        bboxes[:, 1] = ((x1 + x2) / 2) / padded_w
        bboxes[:, 2] = ((y1 + y2) / 2) / padded_h
        bboxes[:, 3] *= 1 / padded_w
        bboxes[:, 4] *= 1 / padded_h

        #bboxes的格式为(category,x,y,w,h)
        targets = torch.zeros((len(bboxes), 6))
        targets[:, 1:] = bboxes
        return img, targets
    
    def collate_fn(self, batch):
        """将数据和标签拼接成batch"""
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        # target [index of imag, category, x, y, w, h]
        targets = [bboxes for bboxes in targets if bboxes is not None]
        # Add sample index to targets
        for i, bboxes in enumerate(targets):
            bboxes[:, 0] = i # 使用索引表示哪些bboxes对应batch中的那张图片 此时bboxes的格式为(index,category,x,y,w,h)
            # print('index',i,'\nbboxes',bboxes)
            # print(np.shape(bboxes))
        targets = torch.cat(targets, 0) #拼接 #TODO 不同标签annotation的box个数不相同
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return imgs, targets
    
    def __len__(self):
        return len(self.ids)
