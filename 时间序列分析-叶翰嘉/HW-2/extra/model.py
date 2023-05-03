import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 搭建ResNet-18模型
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling = False):
        super().__init__()
        self.down_sampling = down_sampling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                                stride = (1 if in_channels == out_channels else 2), padding = 1,
                                bias = False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU())
        ]))
        self.shortcut = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 2, bias = False)),
            ('bn', nn.BatchNorm2d(out_channels))
        ])) if in_channels != out_channels else nn.Sequential()
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.conv2(fx)
        x = self.shortcut(x)
        hx = fx + x
        hx = self.relu2(hx)
        return hx

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
        ]))
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.make_layer(64, 64, down_sampling = False)
        self.layer2 = self.make_layer(64, 128)
        self.layer3 = self.make_layer(128, 256)
        self.layer4 = self.make_layer(256, 512)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 512, out_features = num_classes)

    @staticmethod
    def make_layer(in_channels, out_channels, down_sampling = True):
        layer = nn.Sequential()
        layer.add_module('block1', IdentityBlock(in_channels, out_channels, down_sampling = down_sampling))
        layer.add_module('block2', IdentityBlock(out_channels, out_channels, down_sampling = False))
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
