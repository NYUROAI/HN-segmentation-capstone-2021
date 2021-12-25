import torch
import torchvision
import torchio as tio

class preprocess():
    
    def __init__(self):
        self.training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad((512, 512, 200)),
            tio.transforms.Crop(cropping=(170, 170, 120, 120, 70, 5)),
            tio.Resample((2, 2, 3)),
            tio.CropOrPad((104, 176, 120)),
            tio.ZNormalization(masking_method = tio.ZNormalization.mean)
        ])
        
        self.validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad((512, 512, 200)),
            tio.transforms.Crop(cropping = (170, 170, 120, 120, 70, 5)),
            tio.Resample((2, 2, 3)),
            tio.CropOrPad((104, 176, 120)),
            tio.ZNormalization(masking_method = tio.ZNormalization.mean)
        ])

        self.test_transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad((512, 512, 200)),
            tio.transforms.Crop(cropping = (170, 170, 120, 120, 70, 5)),
            tio.Resample((2, 2, 3)),
            tio.CropOrPad((104, 176, 120)),
            tio.ZNormalization(masking_method = tio.ZNormalization.mean)
        ])
    
    
    def transformation(self, training_subjects, validation_subjects, test_subjects):
        training_set = tio.SubjectsDataset(
            training_subjects, transform = self.training_transform)

        validation_set = tio.SubjectsDataset(
            validation_subjects, transform = self.validation_transform)
        
        test_set = tio.SubjectsDataset(
            test_subjects, transform = self.test_transform)
        
        return training_set, validation_set, test_set