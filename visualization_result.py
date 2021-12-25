from test import test
from preprocess import preprocess
from os import listdir
import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
from util import get_model_and_optimizer

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

best_models = {'Mandible': 'ellisc05_models/11-24-0.5bce-0.5bce-mandible',
               'Parotid_L': 'ellisc05_models/11-27-bce-0.3dice-0.7bce-lrfc0.5-parotidL', 
               'Parotid_R': 'ellisc05_models/11-27-bce-0.3dice-0.7bce-lrfc0.5-parotidR',  
               'BrainStem': 'team_models/BrainStem_4dice_6boundary_state_dict.pth', 
               'SpinalCord': 'team_models/SpinalCord_7bce_3dice_state_dict.pth', 
               'Submandibula_L': 'team_models/Submandibula_L_2dice_8boundary_state_dict.pth', 
               'Submandibula_R': 'team_models/Submandibula_R_6bce_4dice_state_dict.pth', 
               'OpticNerve_L': 'team_models/OpticNerve_L_4bce_6dice_state_dict.pth', 
               'OpticNerve_R': 'team_models/OpticNerve_R_2bce_8dice_state_dict.pth', 
               'OpticChiasm': 'team_models/OpticChiasm_bce_4dice_6bd_state_dict.pth' 
              }

def predict(image_folder):
    
    image_dir = listdir(image_folder)
    predictions = {}
    targets = {}
    
    if 'image.nii.gz' in image_dir:
        
        for organ in best_models.keys():
            if f'mask_{organ}.nii.gz' in image_dir:
                
                print(f'Predicting for {organ}')
                
                organ_subject = tio.Subject(img = tio.ScalarImage(f'{image_folder}/image.nii.gz'), 
                                            mask = tio.LabelMap(f'{image_folder}/mask_{organ}.nii.gz'))
                organ_subject = tio.SubjectsDataset([organ_subject], 
                                                     transform = preprocess().test_transform)
                    
                state_dict = torch.load(f'{best_models[organ]}') 
                loadmodel, _ = get_model_and_optimizer(torch.device('cuda'))
                
                if 'weights' in state_dict.keys():
                    loadmodel.load_state_dict(state_dict['weights'])
                else:
                    loadmodel.load_state_dict(state_dict)
                
                test_obj = test(threshold = 'top_prob', organ = organ)
                
                probability, target, _, _, _ = test_obj.run_test(organ_subject, loadmodel)
                
                predictions[organ] = test_obj.top_probability(probability[0])
                targets[organ] = target[0]                
                
        return organ_subject, predictions, targets 
    
    else:
        raise ValueError('No CT image in this folder')
        
        
def plot(organ_subject, predictions, targets, axslice, color_mapping):
    
    fig, axes = plt.subplots(1, 2, figsize = (18, 12))
    
    axes[0].imshow(np.rot90(organ_subject[0]['img']['data'][0, :, :, axslice]), cmap = 'gray')
    axes[1].imshow(np.rot90(organ_subject[0]['img']['data'][0, :, :, axslice]), cmap = 'gray')
    axes[0].set_title('Prediction')
    axes[1].set_title('Ground Truth')
    
    for organ in predictions.keys():
        
        axes[0].contour(np.rot90(predictions[organ][0, 1, :, :, axslice].cpu().numpy()), 
                        colors = color_mapping[organ], 
                        linewidths = 0.5, 
                        linestyles = 'dotted')
        axes[1].contour(np.rot90(targets[organ][0, 0, :, :, axslice].cpu().numpy()), 
                        colors = color_mapping[organ], 
                        linewidths = 0.5, 
                        linestyles = 'dotted')
        
        
def plot_colortable(colors, title, sort_colors = True, emptycols = 0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize = (width / dpi, height / dpi), dpi = dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize = 24, loc = "left", pad = 10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment = 'left',
                verticalalignment = 'center')

        ax.add_patch(
            Rectangle(xy = (swatch_start_x, y-9), width = swatch_width,
                      height = 18, facecolor = colors[name], edgecolor = '0.7')
        )

    return fig
    
    
    
                            
    
        
        
                        
               