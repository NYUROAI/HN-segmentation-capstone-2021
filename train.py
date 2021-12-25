import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from scipy import stats
from unet import UNet 

from tqdm.notebook import tqdm

# script
from util import prepare_batch, get_model_and_optimizer
import lossOption

import enum
import time
import random
import multiprocessing

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

class train_validate():
    
    def __init__(self, 
                 organ, 
                 save_to, 
                 num_epoch = 8,
                 loss_method = 'single',
                 param = None,
                 scheduled_param = None,
                 used_loss_name = None
                ):
        '''
        organ: string of organ name
        save_to: f string of path
        num_epoch: integer of number of epochs
        loss_method: one from ['single', 'weighted', 'scheduled']
        param: dictionary of {'Name': list of loss names,
                              'Weights': list of loss weights}
        scheduled_param: dictionary of {'First': first combination/single loss before scheduling,
                                        'FirstWeights': list of loss weights,
                                        'Second': second combination/single loss after scheduling,
                                        'SecondWeights': list of loss weights
                                        'Scheduler': scheduling method}
        '''
        
        self.organ = organ
        self.weights_path = save_to
        self.num_epochs = num_epoch
        self.loss_method = loss_method
        #self.loss_candidate = set(['Dice', 'BCE', 'Focal', 'Tversky', 'FocalTversky', 'Boundary', 'HD'])
        
        if self.loss_method is None or self.loss_method not in set(['single', 'weighted', 'scheduled']):
            raise ValueError('type error')
        
        if loss_method == 'single' or loss_method == 'weighted':
            if param is None:
                raise ValueError('parameter not dictionary')
            for elem in param.items():
                if elem[0] not in set(['Name', 'Weights']) or elem[1] is None or len(elem[1]) < 1:
                    raise ValueError('parameter form incorrect')
            self.param = param
        
        if loss_method == 'scheduled':
            if scheduled_param is None:
                raise ValueError('parameter form incorrect')
            standard = set(['First', 'FirstWeights', 'Second', 'SecondWeights', 'Scheduler'])
            if set(scheduled_param.keys()) > standard or len(set(scheduled_param.keys())) != len(standard):
                raise ValueError('parameter form incorrect')
            for elem in scheduled_param.items():
                if elem[1] is None or len(elem[1]) < 1:
                    raise ValueError('parameter missing value')
            self.param = scheduled_param
        

    def run_train_val(self, train, val):
        training_batch_size = 1
        validation_batch_size = 1
        
        training_loader = torch.utils.data.DataLoader(
            train,
            batch_size = training_batch_size,
            shuffle = True,
            num_workers = 2
        )
    
        validation_loader = torch.utils.data.DataLoader(
            val,
            batch_size = validation_batch_size,
            num_workers = 2
        )
    
        model, optimizer = get_model_and_optimizer(device)
        train_losses, val_losses = self.process(self.num_epochs, training_loader, validation_loader, model, optimizer)
    
        checkpoint = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weights': model.state_dict()
        }
        torch.save(checkpoint, self.weights_path)
    
        return model, train_losses, val_losses
    

    def process(self, num_epochs, training_loader, validation_loader, model, optimizer):
        train_losses = []
        val_losses = []
        memorize = [10.000, False]
        if self.loss_method == 'scheduled':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   mode='min', 
                                                                   factor=self.param['Scheduler'][0],
                                                                   patience=self.param['Scheduler'][1])
        else:
            scheduler = None
    
        for epoch_idx in range(1, num_epochs + 1):
            print('Starting epoch', epoch_idx)
            times, epoch_losses, memorize = self.run_epoch(epoch_idx, 
                                                           Action.TRAIN, 
                                                           training_loader, 
                                                           model, 
                                                           optimizer, 
                                                           memorize, 
                                                           scheduler)
            train_losses.append([times, epoch_losses])

            vtimes, vlosses, _ = self.run_epoch(epoch_idx, 
                                                Action.VALIDATE, 
                                                validation_loader, 
                                                model, 
                                                optimizer, 
                                                memorize, 
                                                scheduler)
            val_losses.append([vtimes, vlosses])
    
        return np.array(train_losses), np.array(val_losses)
    
    def run_epoch(self, epoch_idx, action, loader, model, optimizer, memorize, scheduler):
    
        is_training = action == Action.TRAIN
        epoch_losses = []
        times = []
        model.train(is_training)
        flag = memorize[1]
    
        for batch_idx, batch in enumerate(tqdm(loader)):
        
            inputs, targets = prepare_batch(batch, device)
            inputs = inputs.float()
            optimizer.zero_grad()
        
            with torch.set_grad_enabled(is_training):
                logits = model(inputs)
                probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            
                if self.loss_method == 'single':
                    loss_class = self.param['Name'][0]
                    batch_loss = self.param['Weights'][0] * loss_class.__call__(probabilities, targets)
            
                if self.loss_method == 'weighted':
                    batch_loss = torch.Tensor([0]).to(device)
                    for i in range(len(self.param['Weights'])):
                        loss_class = self.param['Name'][i]
                        batch_loss = batch_loss + self.param['Weights'][i] * loss_class.__call__(probabilities, targets)
            
                if self.loss_method == 'scheduled': 
                    batch_loss = torch.Tensor([0]).to(device)
                    if flag:
                        for i in range(len(self.param['SecondWeights'])):
                            batch_loss = batch_loss + self.param['SecondWeights'][i] * self.param['Second'][i].__call__(probabilities, targets)
                    else:
                        for i in range(len(self.param['FirstWeights'])):
                            batch_loss = batch_loss + self.param['FirstWeights'][i] * self.param['First'][i].__call__(probabilities, targets)
            
                if is_training:
                    batch_loss.backward()
                    optimizer.step()
            
                times.append(time.time())
                epoch_losses.append(batch_loss.item())
    
        epoch_losses = np.array(epoch_losses)
    
        if is_training and np.abs(epoch_losses.mean() - memorize[0]) <= 0.002:
            flag = True
        if is_training:
            memorize[0] = epoch_losses.mean()
            memorize[1] = flag
        if not is_training and self.loss_method == 'scheduled':
            scheduler.step(epoch_losses.mean())
        
        print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    
        return times, epoch_losses, memorize 
    