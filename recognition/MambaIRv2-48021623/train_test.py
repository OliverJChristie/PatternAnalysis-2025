"""
train.py
Test training script for SimpleColorCNN, using Caltech-256.
This script is for running LOCALLY (not on the cluster).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import Caltech256

# Import our custom modules
from dataset import ColorizationDataset
from modules import SimpleColorCNN

# --- 1. Configuration ---
EPOCHS = 5         # Caltech-256 is ~30k images, can run more epochs
LEARNING_RATE = 1e-3
BATCH_SIZE = 64     # Adjust this based on your local GPU/CPU
CROP_SIZE = 96      # Using 96x96 crops for the simple test model
DATA_ROOT = "./data"  # Download location
VAL_SPLIT_RATIO = 0.1
SEED = 42

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")


def main():
    # --- 2. Load and Split Data (Caltech-256) ---
    print("Loading Caltech-256 dataset...")
    
    # Download the full dataset to ./data
    # This will be ~3GB. It only downloads the first time.
    try:
        full_dataset = Caltech256(root=DATA_ROOT, download=True)
    except Exception as e:
        print(f"Error downloading Caltech-256: {e}")
        print("Please check your internet connection and disk space.")
        return
        
    print("Dataset download/check complete.")

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
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True # Use 0 workers if you have issues
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"Datasets loaded. Train: {len(train_dataset)}, Val: {len(test_dataset)}")

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = SimpleColorCNN(in_channels=1, out_channels=3).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training & Validation Loop ---
    history = {'train_loss': [], 'val_loss': []}
    print("--- Starting Test Training (on Caltech-256) ---")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for gray_images, color_images in train_pbar:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            predicted_colors = model(gray_images)
            loss = criterion(predicted_colors, color_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray_images, color_images in test_loader:
                gray_images = gray_images.to(device)
                color_images = color_images.to(device)
                predicted_colors = model(gray_images)
                loss = criterion(predicted_colors, color_images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    print("--- Test Training Finished ---")

    # --- 5. Save Model & Plot ---
    model_save_path = "simple_cnn_caltech.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Test model saved to {model_save_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Test Model (Caltech-256) - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("test_loss_plot_caltech.png")
    print("Loss plot saved to test_loss_plot_caltech.png")

if __name__ == "__main__":
    main()