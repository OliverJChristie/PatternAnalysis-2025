"""
modules.py

Contains the model architecture(s).
For testing, we define a SimpleColorCNN
"""

import torch
import torch.nn as nn

class SimpleColorCNN(nn.Module):
    """
    A very simple Fully Convolutional Network for testing the colorization pipeline.
    It takes a 1-channel (grayscale) image and outputs a 3-channel (RGB) image.

    Input: [B, 1, H, W] (normalized to [-1, 1])
    Output: [B, 3, H, W] (normalized to [-1, 1])
    """
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.layers = nn.Sequential(
            # Input: [B, 1, 96, 96]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLu(inplace=True),

            # [B, 32, 96, 96]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)

            # [B, 64, 96, 96]
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLu(inplace=True),

            # [B, 32, 96, 96]
            # Output layer: maps back to 3 channels
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),

            nn.Tanh()
            # Output: [B, 3, 96, 96]
        )
    def forward(self, x):
        return self.layers(x)