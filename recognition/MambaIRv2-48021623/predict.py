"""
predict.py
Loads the trained SimpleColorCNN and runs it on a
full-size image to demonstrate its fully-convolutional nature.
"""
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import SimpleColorCNN

# --- 1. Configuration ---
MODEL_PATH = "models/simple_cnn_caltech.pth"
IMAGE_PATH = "test_images/test_image.jpg"
SAVE_PATH = "predicted_full_size.png"

# --- 2. Load Model ---
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleColorCNN(in_channels=1, out_channels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode
print(f"Model loaded from {MODEL_PATH} on {device}")

try:
    original_pil_img = Image.open(IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"Error: Test image not found at {IMAGE_PATH}")
    print("Please download a test image and save it as 'my_full_size_test_image.jpg'")
    exit()
    
w, h = original_pil_img.size
print(f"Loaded full-size image: {w}x{h}")

# Define transforms for inference
# We DON'T crop here. We just convert to a normalized tensor.
preprocess_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(), # [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]) # [-1, 1]
])

# Create the grayscale input tensor
input_tensor = preprocess_gray(original_pil_img)
input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dim [1, 1, H, W]

# --- 4. Run Prediction ---
print("Running inference on full-size image...")
with torch.no_grad():
    predicted_tensor = model(input_tensor)
print("Inference complete.")

# --- 5. Post-process the Output ---
def denormalize(tensor):
    # De-normalize from [-1, 1] back to [0, 1]
    return (tensor * 0.5) + 0.5

output_tensor = predicted_tensor.squeeze(0).cpu() # Shape [3, H, W]
output_tensor = denormalize(output_tensor).clamp(0, 1)

# Convert tensor to PIL Image
output_img = transforms.ToPILImage()(output_tensor)

# --- 6. Save and Show Results ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_pil_img)
axes[0].set_title(f"Original ({w}x{h})")
axes[0].axis("off")
axes[1].imshow(output_img)
axes[1].set_title(f"Predicted ({output_img.width}x{output_img.height})")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"Prediction saved to {SAVE_PATH}")