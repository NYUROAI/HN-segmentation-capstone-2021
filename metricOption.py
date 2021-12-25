import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from scipy import stats
import nibabel as nib
import medpy

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Dice():
    def __init__(self, epsilon=1e-9):
        self.name = 'Dice Score'
        self.epsilon = epsilon
        self.SPATIAL_DIMENSIONS = 2, 3, 4
    
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        p0 = output
        g0 = target
        p1 = 1 - p0
        g1 = 1 - g0
        
        tp = (p0 * g0).sum(dim=self.SPATIAL_DIMENSIONS)
        fp = (p0 * g1).sum(dim=self.SPATIAL_DIMENSIONS)
        fn = (p1 * g0).sum(dim=self.SPATIAL_DIMENSIONS)
        num = 2 * tp
        denom = 2 * tp + fp + fn + self.epsilon
        
        dice_score = num / denom
        
        return dice_score

class PPV():
    def __init__(self, epsilon=1e-9):
        self.name = 'PPV'
        self.epsilon = epsilon
        self.SPATIAL_DIMENSIONS = 2, 3, 4
        
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        p0 = output
        g0 = target
        p1 = 1 - p0
        g1 = 1 - g0
        
        tp = (p0 * g0).sum(dim=self.SPATIAL_DIMENSIONS)
        fp = (p0 * g1).sum(dim=self.SPATIAL_DIMENSIONS)
        num = tp
        denom = tp + fp + self.epsilon
        
        ppv = num / denom
        
        return ppv

class sensitivity():
    def __init__(self, epsilon=1e-9):
        self.name = 'Sensitivity'
        self.epsilon = epsilon
        self.SPATIAL_DIMENSIONS = 2, 3, 4
        
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        p0 = output
        g0 = target
        p1 = 1 - p0
        g1 = 1 - g0
        
        tp = (p0 * g0).sum(dim=self.SPATIAL_DIMENSIONS)
        fn = (p1 * g0).sum(dim=self.SPATIAL_DIMENSIONS)
        num = tp
        denom = tp + fn + self.epsilon
        
        sen = num / denom
        
        return sen

class HD95():
    def __init__(self):
        self.name = '95% Hausdorff Distance'
        self.SPATIAL_DIMENSIONS = 2, 3, 4
    
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        output = output.squeeze(0).squeeze(0)
        output = output.cpu().numpy()
        
        target = target.squeeze(0).squeeze(0)
        target = target.cpu().numpy()
        
        hdloss = medpy.metric.binary.hd95(output, target)
        
        return torch.Tensor(hdloss)
        