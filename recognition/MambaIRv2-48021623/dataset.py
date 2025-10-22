"""
dataset.py

A wrapper dataset that applies colorization transforms.
It takes a subset of an existing dataset (like Caltech-256)
and returns (grayscale_input, color_target) pairs.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ColorizationDataset(Dataset):
    def __init__(self, subset, split='train', crop_size=96):
        super().__init__()
        self.split = split
        self.subset = subset
        self.crop_size = crop_size
        
        # Define transforms
        # Training: Resize so shortest edge is 128, then take a 96x96 random crop
        self.train_transform = transforms.Compose([
            transforms.Resize(128, antialias=True), # Resize to 128 first
            transforms.RandomCrop(crop_size),      # Take 96x96 random crop
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        
        # Validation: Resize so shortest edge is 128, then take a 96x96 center crop
        self.val_transform = transforms.Compose([
            transforms.Resize(128, antialias=True),
            transforms.CenterCrop(crop_size),
        ])
        
        # Assign paths and transforms based on the split
        if split == 'train':
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform
            
        # Tensor conversions and normalizations
        self.to_tensor = transforms.ToTensor()
        # Normalize to [-1, 1]
        self.normalize_rgb = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize_gray = transforms.Normalize(mean=[0.5], std=[0.5])
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            pil_img, _ = self.subset[idx]
        except Exception as e:
            # Handle potential corrupt images in the dataset
            print(f"Warning: Skipping corrupt image at index {idx}: {e}")
            # Try to load the next image instead
            return self.__getitem__((idx + 1) % len(self))
        
        # Ensure it's RGB (Caltech-256 has some grayscale)
        pil_img = pil_img.convert('RGB')
        
        # Apply train/val transforms (resize, crop, flip)
        aug_pil_img = self.transform(pil_img)
        
        # Create color target tensor
        target_tensor = self.to_tensor(aug_pil_img)
        
        # Create grayscale input tensor
        input_tensor = self.to_grayscale(target_tensor)
        
        # Normalize both to [-1, 1]
        target_norm = self.normalize_rgb(target_tensor)
        input_norm = self.normalize_gray(input_tensor)
        
        return input_norm, target_norm