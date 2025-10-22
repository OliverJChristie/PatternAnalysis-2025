"""
modules.py

Contains the model architecture(s).
For testing, we define a SimpleColorCNN.
"""

import torch
import torch.nn as nn
import timm

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
            # Input: [B, 1, H, W]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # [B, 32, H, W]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # [B, 64, H, W]
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # [B, 32, H, W]
            # Output layer: maps back to 3 channels
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            
            # Final activation: Tanh
            nn.Tanh()
            # Output: [B, 3, H, W]
        )

    def forward(self, x):
        return self.layers(x)

class VimColorizer(nn.Module):
    """
    VimColorizer: A U-Net-style Encoder-Decoder model for colorization.

    Uses a pre-trained Vision Mamba (Vim) model as the encoder.
    A simple convolutional decoder upsamples the features back to a color image.

    Input: [B, 1, 224, 224] (Grayscale, normalized [-1, 1])
    Output: [B, 3, 224, 224] (Color, normalized [-1, 1])
    """
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.encoder = timm.create_model(
            'vim_tiny_patch16_224_bimambav2_final_pool_in1k',
            pretrained=True
        )

        # The econded expects 3-channel input. We'll handle this in the forward pass.
        # Get the number of features from the encoder's output.
        # For vim-tiny, this is 192
        encoder_out_dim = self.encoder.head.in_features

        # Decoder
        # This decoder takes the 14x14 feature map from the encoder and upsamples
        # it back to 224x224

        # Bottleneck feature dimension (D)
        self.bottleneck_dim = encoder_out_dim

        self.decoder = nn.Sequential(
            # Input: [B, 192, 14, 14]

            # Upsample to 28x28
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample to 56x56
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample to 112x112
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Upsample to 224x224
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Final convolution to get 3 output channels
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),

            # Final activation: Tanh to map outputs to [-1, 1]
            nn.Tanh()
        )

    def forward(self, x_gray):
        # x_gray is [B, 1, 224, 224]

        # We repeat the grayscale channel 3 times to fit encoder input
        x_3channel = x_gray.repeat(1, 3, 1, 1)

        features = self.encoder.forward_features(x_3channel) # Shape [B, 197, 192]

        features_no_cls = features[:, 1:, :] # Shape [B, 196, 192]

        B, N, D = features_no_cls.shape
        H = W = int(N**0.5) # 196**0.5=14

        # permute to [B, D, N] then reshape to [B, D, H, W]
        features_map = features_no_cls.permute(0, 2, 1).reshape(B, D, H, W)
        # Shape is now [B, 192, 14, 14]

        # Pass through the decoder
        output = self.decoder(features_map) # Shape [B, 3, 224, 224]

        return output
        
