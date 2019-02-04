import torch
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Proposed model as described in our paper
# Hand and Object Segmentation from Depth Image using Fully Convolutional Network
class FCN_NEW(nn.Module):
    # n is the number of class labels
    # f is the number of channels of filters
    def __init__(self, n=8, f=20):
        super(FCN_NEW, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.conv1 = nn.Conv2d(1, f, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(f, f, kernel_size=5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(f, f, kernel_size=5, padding=2, stride=2)
        self.conv5 = nn.Conv2d(f, f, kernel_size=7, padding=3, stride=2)
        
        self.btnm1 = nn.BatchNorm2d(f)
        self.btnm2 = nn.BatchNorm2d(f)
        self.btnm3 = nn.BatchNorm2d(f)
        self.btnm4 = nn.BatchNorm2d(f)
        self.btnm5 = nn.BatchNorm2d(f)
        
        # For sequential ConvTranpose
        self.tconv6 = nn.ConvTranspose2d(f, f, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv7 = nn.ConvTranspose2d(f, f, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv8 = nn.ConvTranspose2d(f, f, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv9 = nn.ConvTranspose2d(f, f, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.btnm6 = nn.BatchNorm2d(f)
        self.btnm7 = nn.BatchNorm2d(f)
        self.btnm8 = nn.BatchNorm2d(f)
        self.btnm9 = nn.BatchNorm2d(f)

        self.conv10 = nn.Conv2d(f, n, kernel_size=1)

    def forward(self, x):     
        x0 = self.dropout(x)  

        # Note: bs:batchsize, H: height of image, W: width of image
        x1 = F.relu(self.btnm1(self.conv1(x0))) # Block 1: bs x 1 x H x W --> bs x f x H x W
        x2 = F.relu(self.btnm2(self.conv2(x1))) # Block 2: bs x f x H x W --> bs x f x H/2 x W/2
        x3 = F.relu(self.btnm3(self.conv3(x2))) # Block 3: bs x f x H/2 x W/2 --> bs x f x H/4 x W/4 
        x4 = F.relu(self.btnm4(self.conv4(x3))) # Block 4: bs x f x H/4 x W/4 --> bs x f x H/8 x W/8
        x5 = F.relu(self.btnm5(self.conv5(x4))) # Block 5: bs x f x H/8 x W/8 --> bs x f x H/16 x W/16
        
        x6 = F.relu(self.btnm6(self.tconv6(x5)) + x4) # Block 6: bs x f x H/16 x W/16 --> bs x f x H/8 x W/8
        x7 = F.relu(self.btnm7(self.tconv7(x6)) + x3) # Block 7: bs x f x H/8 x W/8 --> bs x f x H/4 x W/4
        x8 = F.relu(self.btnm8(self.tconv8(x7)) + x2) # Block 8: bs x f x H/4 x W/4 --> bs x f x H/2 x W/2
        x9 = F.relu(self.btnm9(self.tconv9(x8)) + x1) # Block 9: bs x f x H/2 x W/2 --> bs x f x H x W

        x10 = self.conv10(x9) # Block 10: bs x n x H x W

        return x10


# The FCN model below is adapted from the paper by Taylor et al.
# Articulated Distance Fields for Ultra-fast Tracking of Hands Interacting
# http://doi.acm.org/10.1145/31H/16W/80.31H/16853
class FCN_ADF(nn.Module):
    # n is the number of class labels
    # f is the number of channels of filters
    def __init__(self, n=8, f=10): 
        super(FCN_ADF, self).__init__()
        self.dropout = nn.Dropout(0.1)

        self.conv1 = nn.Conv2d(1, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f, f, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(f, f, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(f, f, kernel_size=7, padding=3)
        
        self.btnm1 = nn.BatchNorm2d(f)
        self.btnm2 = nn.BatchNorm2d(f)
        self.btnm3 = nn.BatchNorm2d(f)
        self.btnm4 = nn.BatchNorm2d(f)
        self.btnm5 = nn.BatchNorm2d(f)

        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(2,2)
        
        self.conv6a = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.tconv6b = nn.ConvTranspose2d(f, f, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv6c = nn.ConvTranspose2d(f, f, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.tconv6d = nn.ConvTranspose2d(f, f, kernel_size=3, stride=8, padding=0, output_padding=5)
        self.tconv6e = nn.ConvTranspose2d(f, f, kernel_size=3, stride=16, padding=0, output_padding=13)

        self.btnm7 = nn.BatchNorm2d(f*5)
        self.conv7 = nn.Conv2d(f*5, n, kernel_size=1)

    def forward(self, x):    
        x0 = self.dropout(x)   
        
        x1 = self.btnm1(F.relu(self.conv1(x0)))             # Block 1: bs x 1 x H x W --> bs x f x H x W
        x2 = self.btnm2(F.relu(self.conv2(self.pool1(x1)))) # Block 2: bs x f x H x W --> bs x f x H/2 x W/2
        x3 = self.btnm3(F.relu(self.conv3(self.pool2(x2)))) # Block 3: bs x f x H/2 x W/2 --> bs x f x H/4 x W/4 
        x4 = self.btnm4(F.relu(self.conv4(self.pool3(x3)))) # Block 4: bs x f x H/4 x W/4 --> bs x f x H/8 x W/8
        x5 = self.btnm5(F.relu(self.conv5(self.pool4(x4)))) # Block 5: bs x f x H/8 x W/8 --> bs x f x H/16 x W/16
        
        x6a = self.conv6a(x1)  # Block 6a: bs x f x H x W --> bs x f x H x W
        x6b = self.tconv6b(x2) # Block 6b: bs x f x H/2 x W/2 --> bs x f x H x W
        x6c = self.tconv6c(x3) # Block 6c: bs x f x H/4 x W/4 --> bs x f x H x W
        x6d = self.tconv6d(x4) # Block 6d: bs x f x H/8 x W/8 --> bs x f x H x W
        x6e = self.tconv6e(x5) # Block 6e: bs x f x H/16 x W/16 --> bs x f x H x W

        x7 = torch.cat((x6a,x6b,x6c,x6d,x6e), dim=1) # Block 7: bs x 5f x H x W (dim=1 as dim0 is the batchsize)
        x7 = self.btnm7(F.relu(x7))
        x7 = self.conv7(x7)     # Block 7: bs x n x H x W

        return x7
