"""
train.py
Distributed training script for VimColorizer on Caltech-256
Designed to be run on the Rangpur cluster using torchrun.
e.g.: torchrun --nproc_per_node=4 train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import Caltech256

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import ColorizationDataset
from modules import VimColorizer

# Configuration
EPOCHS = 20

DECODER_LR = 1e-4
ENCODER_LR = 1e-6

BATCH_SIZE = 32
CROP_SIZE = 224
DATA_ROOT = "./data"
VAL_SPLIT_RATIO = 0.1
SEED = 42
MODEL_SAVE_PATH = "models/vim_colorizer_caltech.pth"
PLOT_SAVE_PATH = "plots/vim_loss_plot_caltech.png"

def setup_ddp():
    """
    Initializes the DDP process group
    """
    dist.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup_ddp():
    """ Cleans up teh DDP proces group. """
    dist.destroy_process_group()

def main():
    rank, local_rank = setup_ddp()
    is_main_process = (rank == 0)

    if is_main_process:
        print(f"--- Starting DDP Training on {dist.get_world_size()} GPUs ---")
        print(f"--- Using device: cuda:{local_rank} ---")
    
    if is_main_process:
        print("Rank 0: Loading/Downloading Caltech-256 dataset...")
        try:
            full_dataset = Caltech256(root=DATA_ROOT, download=True)
            print("Rank 0: Dataset download/check complete.")
        except Exception as e:
            print(f"Rank 0: Error downloading Caltech-256: {e}")
            return
    
    # Wait for rank 0 to finish downloading/checking
    dist.barrier()

    if not is_main_process:
        full_dataset = Caltech256(root=DATA_ROOT, download=False)
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Wrap the subsets in our ColorizationDataset
    train_dataset = ColorizationDataset(
        train_subset, split='train', crop_size=CROP_SIZE
    )
    test_dataset = ColorizationDataset(
        val_subset, split='val', crop_size=CROP_SIZE
    )

    # DDP Dataloaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=4, pin_memory=True,
        shuffle=False # Sampler handles shuffling
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        sampler=test_sampler, num_workers=4, pin_memory=True,
        shuffle=False
    )

    if is_main_process:
        print(f"Datasets loaded. Total Train: {len(train_dataset)}, Total Val: {len(test_dataset)}")
        print(f"Batch size per GPU: {BATCH_SIZE}. Total batch size: {BATCH_SIZE * dist.get_world_size()}")
    
    model = VimColorizer(in_channels=1, out_channels=3).to(local_rank)

    model=DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()

    # Create parameter groups for differential learning rates
    param_groups = [
        {
            'params': model.module.encoder.parameters(),
            'lr': ENCODER_LR
        },
        {
            'params': model.module.decoder.parameters(),
            'lr': DECODER_LR
        }
    ]

    optimizer = optim.Adam(param_groups)

    if is_main_process:
        print(f"Optimizer set up with differential LRs:")
        print(f" Encoder (Vim) LR: {ENCODER_LR}")
        print(f" Decoder LR: {DECODER_LR}")
    
    history = {'train_loss': [], 'val_loss': []}
    if is_main_process:
        print(f"--- Starting Traning on {EPOCHS} Epochs ---")
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        model.train()

        if is_main_process:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        else:
            train_pbar = train_loader
        
        train_loss = 0.0
        for gray_images, color_images in train_pbar:
            gray_images = gray_images.to(local_rank)
            color_images = color_images.to(local_rank)

            predicted_colors = model(gray_images)
            loss = criterion(predicted_colors, color_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if is_main_process:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Average loss across all batches (on this GPU)
        avg_train_loss_gpu = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray_images, color_images in test_loader:
                gray_images = gray_images.to(local_rank)
                color_images = color_images.to(local_rank)
                predicted_colors = model(gray_images)
                loss = criterion(predicted_colors, color_images)
                val_loss += loss.item()
        
        avg_val_loss_gpu = val_loss / len(test_loader)

        # Aggregate losses from all GPUs
        train_loss_tensor = torch.tensor(avg_train_loss_gpu).to(local_rank)
        val_loss_tensor = torch.tensor(avg_val_loss_gpu).to(local_rank)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)

        avg_train_loss = train_loss_tensor.item()
        avg_val_loss = val_loss_tensor.item()

        if is_main_process:
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
    
    if is_main_process:
        print("--- Training Finished ---")
    
    if is_main_process:
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

        plt.figure(figsize=(10,5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('VimColorizer (Caltech-256) - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_SAVE_PATH)
        print(f"Loss plot saved to {PLOT_SAVE_PATH}")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()