import os
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.dataset import Dataset
'''Data Transformation refered: 
https://pytorch.org/vision/0.9/transforms.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder
'''
class ImageGTZANDataset(Dataset):
    def __init__(self, data_dir, subdir, transform=True):
        """
        data_dir (str): Path to data containing images data. 
        subdir: subdir of data_dir for either test, train or val.
        """
        root_dir = os.path.join(data_dir, subdir)
        if transform:
            
            transformation = transforms.Compose([
                 transforms.Lambda(lambda img: img.convert('RGB')),
                 transforms.Resize((224, 224)),  # The size of image is 315 *217 checked in the folder,so Resize images if not already in size of 224*224
                 transforms.ToTensor(),# Convert the image to a PyTorch tensor, (C, H, W) (3, 224,224)
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # It has three channels after converting to RGB to Normalize the dataset, since the dataset are images of range [0, 1], so we can transform them to Tensors of normalized range [-1, 1] for three channels.
                ])

        
        # Initialize the ImageFolder dataset, which  loads images from a directory this needs the images are organized in subdirectories, where each subdirectory represents a class, and the images within each subdirectory belong to that class
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transformation)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)



