#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodels
from torchvision.transforms import Resize 

#From "Federated Visual Classification with Real-World Data Distribution"
class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()

        if(args.norm == "None"):
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.ReLu(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        elif(args.norm == "batch_norm"):
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
        elif(args.norm == "group_norm"):
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5),
                nn.GroupNorm(int(64/16), 64), #16 channels per group suggested by paper
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5),
                nn.GroupNorm(int(64/16), 64), #16 channels per group suggested by paper
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class mobilenet_v3(nn.Module):
    def __init__(self, args):
        super(mobilenet_v3, self).__init__()

        self.resize = Resize(224)

        self.model = getattr(tvmodels, f'mobilenet_v3_{args.variant}')(args.pretrained)

        self.model.classifier[0] = nn.Linear(self.model.classifier[0].in_features, self.model.classifier[0].out_features)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, args.num_classes)

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        return x

# From "Communication-Efficient Learning of Deep Networks from Decentralized Data" (ported from 2016 TensorFlow CIFAR-10 tutorial)
class cnn_cifar10(nn.Module):
    def __init__(self, num_classes, num_channels, model_args):
        super(cnn_cifar10, self).__init__()

        self.resize = Resize((24, 24))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001/9),
            nn.ZeroPad2d((0, 1, 0, 1)), # Equivalent of TensorFlow padding 'SAME' for MaxPool2d
            nn.MaxPool2d(3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x