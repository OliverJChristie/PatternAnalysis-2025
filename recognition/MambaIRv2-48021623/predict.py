import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from modules import SimpleColorCNN

MODEL_PATH = "models/simple_cnn_test.pth"
IMAGE_PATH = "test_images/test_image.jpg"
IMG_SIZE = 96

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleColorCNN(in_channels=1, out_channels=3)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {MODEL_PATH} on {device}")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(), # Converts to [0, 1] range
    transforms.Grayscale(num_output_channels=1), # Convert to grayscale
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
])

# Load the original color image for comparison
original_pil_img = Image.open(IMAGE_PATH).convert("RGB")
original_pil_img = original_pil_img.resize((IMG_SIZE, IMG_SIZE))

# Process the image to create the grayscale input tensor
input_tensor = preprocess(original_pil_img)
input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dim [1, 1, H, W]

print(f"Loaded and preprocessed image from {IMAGE_PATH}")

# --- 3. Run Prediction ---
with torch.no_grad(): # Disable gradient calculation for inference
    predicted_tensor = model(input_tensor)

# --- 4. Post-process the Output ---
# De-normalize from [-1, 1] back to [0, 1]
def denormalize(tensor):
    return (tensor * 0.5) + 0.5

# Get the first (and only) image from the batch
output_tensor = predicted_tensor.squeeze(0).cpu() # Shape [3, H, W]
output_tensor = denormalize(output_tensor)

# Convert tensor to PIL Image for saving/showing
output_img = transforms.ToPILImage()(output_tensor)

# Prepare original and grayscale for plotting
grayscale_pil_img = transforms.ToPILImage()(denormalize(input_tensor.squeeze(0).cpu()))

# --- 5. Show Results ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(grayscale_pil_img, cmap='gray')
axes[0].set_title("Input (Grayscale)")
axes[0].axis("off")

axes[1].imshow(output_img)
axes[1].set_title("Prediction (SimpleCNN)")
axes[1].axis("off")

axes[2].imshow(original_pil_img)
axes[2].set_title("Ground Truth (Original)")
axes[2].axis("off")

plt.savefig("plots/simple_prediction.png")
print("Prediction saved to plots/simple_prediction.png")