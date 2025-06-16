import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CircularMaskedConv(nn.Module):
    def __init__(self):
        super().__init__()
        size = 17
        radius = 7
        center = (size - 1) / 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = np.where(dist <= radius, 1.0, -1.0).astype(np.float32)
        weight = torch.from_numpy(mask)
        weight /= weight.sum()
        weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, 17, 17]

        self.conv = nn.Conv2d(1, 1, kernel_size=17, padding=8, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

# --- Load images ---
paths = [
    "/mnt/h/latest_real_data/real_data/real/x0.00_y0.00.jpg",
    "/mnt/h/newlight/main_light/x0.00_y0.00.jpg",
    "/mnt/h/newlight/main_light/x1.12_y-1.16.jpg"
]

images = []
processed = []

model = CircularMaskedConv()
model.eval()
clahe = cv2.createCLAHE(clipLimit=5)
for path in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    images.append(img)

    # Convert to tensor and normalize to [0, 1]
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        result = model(tensor).squeeze()

    # Normalize result to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    processed.append(result.numpy())

# --- Plot ---
titles = ["Original Light", "New Light", "Latest Light"]
plt.figure(figsize=(15, 6))

for i in range(3):
    # Row 1: Original
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"{titles[i]} - Original")
    plt.axis('off')

    # Row 2: Processed
    plt.subplot(2, 3, i + 4)
    plt.imshow(processed[i], cmap='gray')
    plt.title(f"{titles[i]} - Filtered")
    plt.axis('off')

plt.tight_layout()
plt.show()
