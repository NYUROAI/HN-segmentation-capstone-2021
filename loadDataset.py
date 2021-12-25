# Last modified: December 2, 2021

from pathlib import Path

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from os import listdir
from os.path import isfile, join
import nibabel as nib


class subjectLoader(Dataset):
    
    def __init__(self, 
                 dataset_dir_name='/gpfs/data/yuanlab/projects/HN_autocontour/data/processed/', 
                 mask_name='mask_Mandible', 
                 training_split_ratio=0.7):
        
        self.dataset_dir_name = dataset_dir_name
        self.folder_list = listdir(dataset_dir_name)
        self.mask_name = mask_name
        
        self.extract_dir_name(self.mask_name)
        
        self.training_split_ratio = training_split_ratio
    
    def __len__(self):
        return len(self.img_paths) - len(self.sample_filter())
    
    
    def __getitem__(self):
     
        subjects = []
        count = 0
        for (image_path, label_path) in zip(self.img_paths, self.lb_paths):
            if count not in self.sample_filter():
                subject = tio.Subject(
                    img=tio.ScalarImage(image_path),
                    mask=tio.LabelMap(label_path)
                )
                subjects.append(subject)
            count += 1
        dataset = tio.SubjectsDataset(subjects)
        
        #--------train val test split---------#
        validation_split_ratio = (1 - self.training_split_ratio) / 2
        num_subjects = len(dataset)
        
        num_training_subjects = int(self.training_split_ratio * num_subjects)
        num_validation_subjects = int(validation_split_ratio * num_subjects)
        num_test_subjects = num_subjects - num_training_subjects - num_validation_subjects
        num_split_subjects = num_training_subjects, num_validation_subjects, num_test_subjects
        training_subjects, validation_subjects, test_subjects = torch.utils.data.random_split(subjects, num_split_subjects)
        
        return training_subjects, validation_subjects, test_subjects
    
        
    def extract_dir_name(self, mask_name):
        dataset_dir = Path(self.dataset_dir_name)
        
        image_paths = []
        label_paths = []
        
        for f_name in self.folder_list:
            name_list = listdir(self.dataset_dir_name+f_name)
            if 'image.nii.gz' in name_list and '{n}.nii.gz'.format(n=mask_name) in name_list:
                images_dir = dataset_dir / '{}/image.nii.gz'.format(f_name)
                image_paths.append(images_dir)
                label_dir = dataset_dir / '{f}/{n}.nii.gz'.format(f=f_name, n=mask_name)
                label_paths.append(label_dir)
        
        self.img_paths = image_paths
        self.lb_paths = label_paths
    
    
    def sample_filter(self):
        '''
        Can be further improved. Get rid of distorted image in NYU Langone dataset.
        '''
        return set([7, 15, 30, 167, 208, 257, 262, 266, 295])
