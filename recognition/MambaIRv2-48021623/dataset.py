"""
dataset.py

This file defines the PyTorch Dataset for the colorization task.
It uses the STL-10 dataset, which it can download automatically.
- It splits the data into 'train' and 'test' sets.
- It applies augmentations (flipping, jitter) to the 'train' set only.
- It returns a (grayscale_image, color_image) pair for training.
- Both tensors are normalized to the range [-1, 1].
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import STL10
import torchvision.transforms as transforms

class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for image colorization.

    Loads the STL-10 dataset, applies augmentations to the training split,
    and returns a (grayscale_input, color_target pair).
    """
    def __init__(self, root, split='train', download=True, img_size=96, transform=None):
        """
        Args:
            root (str): Directory where the dataset will be stored.
            split (str): 'train" or 'test' to load the respective split.
            download (bool): If True, downloads the dataset if not found.
            img_size (int): The target size (H and W) to resize all images to.
                            STL-10 images are 96x96 by default.
            transform (callable, optional): Optional override for the base transform.
        """
        super().__init__()
        self.split = split

        # Download/Load the STL-10 dataset
        self.stl_data = STL10(root=root, split=split, download=download)

        if transform is None:
            if split == 'train':
                self.base_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size), antialias=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
                ])
            else:
                self.base_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size), antialias=True)
                ])
        else:
            self.base_transform = transform
        
        self.to_tensor = transforms.ToTensor()

        self.normalize_rgb = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        self.normalize_gray = transforms.Normalize(mean=[0.5], std=[0.5])

        self.to_grayscale = transforms.Grayscale(num_output_channels=1)
    
    def __len__(self):
        """ Returns the total number of images in the split. """
        return len(self.stl_data)
    
    def __getitem__(self, idx):
        """
        Fetches the image at the given index, processes it, and returns the
        (grayscale_input, color_target) pair.
        """

        pil_img, _ = self.stl_data[idx]

        aug_pil_img = self.base_transform(pil_img)

        target_tensor = self.to_tensor(aug_pil_img)

        input_tensor = self.to_grayscale(target_tensor)

        target_norm = self.normalize_rgb(target_tensor)
        input_norm = self.normalize_gray(input_tensor)
        
        # input_norm shape: [1, H, W]
        # target_norm shape: [3, H, W]
        return input_norm, target_norm