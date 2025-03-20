import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from monai.networks.nets import DenseNet, UNet, ResNet
from monai.transforms import Compose, ToTensor
from monai.inferers import sliding_window_inference

# Modify ResNet model for binary classification
class ResNetBinary(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNetBinary, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18
        # Modify the last fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Convert image to torch.uint8
        x = x.type(torch.uint8)
        return self.model(x.float())


class BinaryDenseNet(DenseNet):
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=1):
        super().__init__(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                         block_config=(6, 12, 24, 16), init_features=64, growth_rate=32)

    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x)  # Sigmoid activation to get binary output