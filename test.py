import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from unet import UNet 

from tqdm.notebook import tqdm

# script
from util import prepare_batch, get_model_and_optimizer
import metricOption

import pickle

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class test():
    
    def __init__(self, threshold = 'top_prob', organ='Mandible'):
        
        self.organ = organ
        self.threshold = threshold
    
    def run_test(self, test_set, model):
        
        test_batch_size = 1
        loader = torch.utils.data.DataLoader(
            test_set,
            batch_size = test_batch_size,
            num_workers = 2
        )    
        
        proba = []
        targ = []
        dice = []
        ppv = []
        sensi = []
    
        for batch_idx, batch in enumerate(tqdm(loader)):
            inputs, targets = prepare_batch(batch, device)
        
            with torch.no_grad():
                logits = model(inputs)
                probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
                if self.threshold == 'top_prob':
                    probabilities = self.top_probability(probabilities)

                batch_score = metricOption.Dice().__call__(probabilities, targets)
                ppv_score = metricOption.PPV().__call__(probabilities, targets)
                sen_score = metricOption.sensitivity().__call__(probabilities, targets)
           
                dice.append(batch_score.item())
                proba.append(probabilities)
                ppv.append(ppv_score.item())
                sensi.append(sen_score.item())
                targ.append(targets)
            
        dice = np.array(dice)
        print(f'dice mean: {dice.mean():0.3f}')
        print(f'dice std: {dice.std():0.3f}')
        
        ppv = np.array(ppv)
        print(f'ppv mean: {ppv.mean():0.3f}')
        print(f'ppv std: {ppv.std():0.3f}')
        
        sensi = np.array(sensi)
        print(f'sensitivity mean: {sensi.mean():0.3f}')
        print(f'sensitivity std: {sensi.std():0.3f}')
        
        return proba, targ, dice, ppv, sensi
        
    def top_probability(self, probabilities):
    
        with open('organ_mask_proportion_full.pkl', 'rb') as f:
            organ_mask_proportion = pickle.load(f)
        omp = np.array(organ_mask_proportion[self.organ])
        threshold = np.percentile(omp, 95)
    
        prob = probabilities[0, 1, :, :, :]
        total_size = 104 * 176 * 120
        proportion = int(threshold * total_size)
    
        value, index = torch.topk(prob.flatten(), proportion)
        nonzero_index = np.array(np.unravel_index(index[value > 0].cpu().numpy(), prob.shape))
    
        output = torch.zeros(probabilities.shape)
        output[0, 1, :, :, :][nonzero_index] = 1
    
        return output.float().to(device)