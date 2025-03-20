import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNetPlusPlus, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.bridges = nn.ModuleList()

        # Downward path
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Upward path and bridges
        for i in range(len(features) - 1):
            self.ups.append(nn.ModuleList())
            self.bridges.append(nn.ModuleList())
            for j in range(len(features) - i - 1):
                self.ups[i].append(UpConv(features[j + i + 1], features[j + i]))
                self.bridges[i].append(ConvBlock(features[j + i] * (2 if j != 0 else 1), features[j + i]))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Upward path
        for i in range(len(self.ups)):
            for j in range(len(self.ups[i])):
                skip = skip_connections[-(j+2)]
                x = self.ups[i][j](x)
                x = torch.cat((x, skip), dim=1)
                x = self.bridges[i][j](x)

        # Final 1x1 convolution
        x = self.final_conv(x)
        return torch.sigmoid(x)  # Using sigmoid for binary classification