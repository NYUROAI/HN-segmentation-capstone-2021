# model, batch, plot
import torch
import torchvision
import torchio as tio
from unet import UNet
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm


def prepare_batch(batch, device):
    inputs = batch['img']['data'].to(device)
    targets = (batch['mask']['data'] // 255).to(device)
    return inputs, targets

def get_model_and_optimizer(device):
    model = UNet(in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=4,
        out_channels_first_layer=32,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU'
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    return model, optimizer

def plot_times(axis, losses, label):
    times, losses = losses.transpose(1, 0, 2)
    times = [datetime.fromtimestamp(x) for x in times.flatten()]
    axis.plot(times, losses.flatten(), label=label)

def plot_loss_time(train_losses, val_losses):    
    fig, ax = plt.subplots()
    plot_times(ax, train_losses, 'Training')
    plot_times(ax, val_losses, 'Validation')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Loss')
    ax.set_title('Training with whole images')
    ax.legend()
    fig.autofmt_xdate()
    
def mask_proportion(data, structures):
    """
    Compute the proportion of masks for each organ in the data.
    """
    
    total_size = 104 * 176 * 120 # data have this size after preprocessing
    organ_mask_proportion = {}
    
    for organ in structures:
        print(f'computing for {organ}....')
        mask_proportion = []
        
        for img in tqdm(data):
            proportion = (img['mask']['data'] // 255).sum() / total_size # masks are 0 and 255, not 0 and 1 in data
            mask_proportion.append(proportion)
    
    organ_mask_proportion[organ] = mask_proportion
    
    return organ_mask_proportion