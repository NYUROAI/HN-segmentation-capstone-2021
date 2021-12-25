# HN-segmentation-capstone-2021

NYU Center of Data Science Capstone Project (Fall 2021) by Claire Ellison-Chen, Kallen Xu, and Yupei Zhou.

[loadDataset.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/loadDataset.py) loads the NYU Langone dataset on Bigpurple and splits it into training/validation/test dataset.

[preprocess.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/preprocess.py) applies necessary preprocessing steps including cropping/padding, resampling and z-normalization to the data.

[train.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/train.py) and [test.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/test.py) contains scripts to train and test the 3D Unet model. 

[lossOption.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/lossOption.py) contains several loss functions for model training. 

[metricOption.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/metricOption.py) contains several metrics for model evaluation. 

[visualization_result.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/visualization_result.py) includes python scripts for visualization. 

[util.py](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/util.py) contains other relevant python scripts. 

[example_for_script.ipynb](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/example_for_script.ipynb) is an end-to-end example of loading and preprocessing data, training the model and testing the model.

[patient_eval.ipynb](https://github.com/NYUROAI/HN-segmentation-capstone-2021/blob/main/patient_eval.ipynb) creates visualization using trained models on patient data. 
