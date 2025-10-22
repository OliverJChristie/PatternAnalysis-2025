"""
train.py

Main script for training the test model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ColorizationDataset
from modules import SimpleColorCNN

# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
IMG_SIZE = 96
DATA_ROOT = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

def main():
    print("Loading datasets...")
    train_dataset = ColorizationDataset(
        root=DATA_ROOT,
        split='train',
        download=True,
        img_size=IMG_SIZE
    )

    test_dataset = ColorizationDataset(
        root=DATA_ROOT,
        split='test',
        download=True,
        img_size=IMG_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Datasets loaded.")

    model = SimpleColorCNN(in_channels=1, out_channels=3).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}

    print("--- Starting Test Training ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss =0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for gray_images, color_images in train_pbar:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)

            # Forward pass
            predicted_colors = model(gray_images)

            # Calculate loss
            loss = criterion(predicted_colors, color_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for gray_images, color_images in val_pbar:
                gray_images = gray_images.to(device)
                color_images = color_images.to(device)
                
                predicted_colors = model(gray_images)
                loss = criterion(predicted_colors, color_images)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(test_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1} / {EPOCHS} -"
              f"Train Loss: {avg_train_loss:.4f},"
              f"Val Loss: {avg_val_loss:.4f}")
    print("--- Test Training Finished ---")

    model_save_path = "models/simple_cnn_test.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Test model saved to {model_save_path}")

    plt.figure(figsize=(10,5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Test Model - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/test_loss_plot.png")
    print("Loss plot saved to plots/test_loss_plot.png")

if __name__ == "__main__":
    main()