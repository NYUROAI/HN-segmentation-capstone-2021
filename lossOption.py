import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from scipy import stats 
import nibabel as nib

from scipy.ndimage import distance_transform_edt as eucl_distance
from skimage import segmentation as skimage_seg
import kornia

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Dice(nn.Module):
    def __init__(self, epsilon=1e-9):
        self.name = 'Dice'
        self.epsilon = epsilon
        self.SPATIAL_DIMENSIONS = 2, 3, 4
    
    def get_dice_score(self, output, target):
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

    def __call__(self, output, target):
        return 1 - self.get_dice_score(output, target)

    
class BCE(nn.Module):
    def __init__(self):
        self.name = 'BCE'

    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        
        bceloss = nn.BCELoss()
        loss = bceloss(output.float(), target.float())
        
        return loss

    
class Focal(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        self.name = 'Focal'
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
        
        loss = nn.BCELoss()
        bce_loss = loss(output.float(), target.float())
        bce_exp = torch.exp(-bce_loss)
        
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce_loss          
        
        return focal_loss

    
class Tversky(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, beta=0.3, smooth=1):
        self.name = 'Tversky'
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.smooth = smooth
        self.SPATIAL_DIMENSIONS = 2, 3, 4

    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]

        #True Positives, False Positives & False Negatives
        TP = (output * target).sum(dim=self.SPATIAL_DIMENSIONS)    
        FP = ((1 - target) * output).sum(dim=self.SPATIAL_DIMENSIONS)
        FN = (target * (1 - output)).sum(dim=self.SPATIAL_DIMENSIONS)
       
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)  
        
        return 1 - Tversky

    
class FocalTversky(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, beta=0.3, smooth=1):
        self.name = 'FocalTversky'
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.smooth = smooth
        self.SPATIAL_DIMENSIONS = 2, 3, 4
    
    def __call__(self, output, target):     
        output = output[:, torch.LongTensor([1]), :, :, :]

        #True Positives, False Positives & False Negatives
        TP = (output * target).sum(dim=self.SPATIAL_DIMENSIONS)    
        FP = ((1 - target) * output).sum(dim=self.SPATIAL_DIMENSIONS)
        FN = (target * (1 - output)).sum(dim=self.SPATIAL_DIMENSIONS)

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)  
        FocalTversky = (1 - Tversky) ** self.gamma
        
        return FocalTversky

    
class Boundary(nn.Module):
    def __init__(self):
        self.name = 'Boundary'
        
    def __call__(self, output, target):
        out_shape = output.shape
        dist_map = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(1, out_shape[1]):
                posmask = target[b]
                negmask = 1-posmask
                posdis = eucl_distance(posmask.cpu().numpy())
                negdis = eucl_distance(negmask.cpu().numpy())
                boundary = skimage_seg.find_boundaries(posmask.cpu().numpy(), mode='inner')
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                dist_map[b][c] = sdf
    
        dist_map = torch.from_numpy(dist_map).to(device)
        output = output[:, torch.LongTensor([1]), :, :, :]
        dist_map = dist_map[:, torch.LongTensor([1]), :, :, :]
    
        multipled = torch.einsum('bkxyz, bkxyz->bkxyz', output, dist_map)
        bd_loss = multipled.mean()
        
        return bd_loss
    
    
class Hausdorff_Loss(nn.Module):
    def __init__(self, alpha=0.2, k=10, reduction='mean'):
        self.name = 'HD'
        self.alpha = alpha
        self.k = k
        self.reduction = reduction
        
    def __call__(self, output, target):
        output = output[:, torch.LongTensor([1]), :, :, :]
  
        HDloss=kornia.losses.HausdorffERLoss3D(alpha=self.alpha, k=self.k, reduction=self.reduction)
        loss = HDloss(output, target)
        return loss
