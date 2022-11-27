import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
from utils import class2one_hot,one_hot2dist
from functools import partial
from operator import itemgetter
from torchvision import transforms

def dist_map_transform(resolution, K):
    return transforms.Compose([
        gt_transform(K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

def gt_transform(K):
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])
