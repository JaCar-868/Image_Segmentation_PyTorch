# PyTorch Image Segmentation

This repository contains code for image segmentation using PyTorch. The model implemented is a U-Net architecture with a ResNet18 backbone.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Image segmentation is a crucial task in computer vision where the goal is to assign a class label to each pixel in the image. This repository provides an implementation of a U-Net model with a ResNet18 backbone for performing image segmentation tasks.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- other necessary libraries (specified in `requirements.txt`)

## Installation

1. Clone this repository:
    
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    

2. Create a virtual environment and activate it:
    
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    

3. Install the required libraries:
   
    pip install -r requirements.txt
   

## Usage

1. Data Preparation

Download the VOCSegmentation dataset:

from torchvision.datasets import VOCSegmentation

train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transforms, target_transform=transforms)
val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transforms, target_transform=transforms)

2. Model Definition
   
The ResNetUNet class is defined to create the segmentation model with a ResNet18 backbone:

import torch.nn as nn
import torchvision

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.conv_up3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        upconv4 = self.upconv4(layer4)
        upconv4 = torch.cat([upconv4, layer3], dim=1)
        upconv4 = self.conv_up3(upconv4)

        upconv3 = self.upconv3(upconv4)
        upconv3 = torch.cat([upconv3, layer2], dim=1)
        upconv3 = self.conv_up2(upconv3)

        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, layer1], dim=1)
        upconv2 = self.conv_up1(upconv2)

        upconv1 = self.upconv1(upconv2)
        upconv1 = torch.cat([upconv1, layer0], dim=1)

        out = self.conv_last(upconv1)
        return out
        
3. Training
To train the model, use the following code:

import torch
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = ResNetUNet(n_classes=21)  # 21 classes for VOC dataset
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation step can be added here

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
4. Evaluation

To evaluate the model, use the validation dataset and calculate relevant metrics.

## Acknowledgments
This implementation is based on the U-Net architecture and uses ResNet18 as the backbone.
The dataset used is Pascal VOC 2012.

## License
This project is licensed under the MIT License - see the LICENSE file for details
