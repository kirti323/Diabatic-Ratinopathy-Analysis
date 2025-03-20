from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import os
import torch
from torchvision import transforms


class DRAC_Test_Loader(Dataset):
    def __init__(self, data = [], data_names = [], transform=None):
        # Intialize the basic variables
        self.data_loc = data
        self.image_names = data_names
        self.transform = transform
        
        # Load the data
        self.images = self.load_data(self.data_loc)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image_name = self.image_names[idx]
        
        # Apply the transformations
        if self.transform:
            # Apply a ToTensor transformation
            image = self.transform(image)
        
        return {'image': image}
    
    def load_data(self, data):
        # Load data from "data" list, which is a list of paths to the images
        images = []
        
        for file in data:
            image = sitk.ReadImage(file)
            image = sitk.GetArrayFromImage(image)
            images.append(image)
        
        return images