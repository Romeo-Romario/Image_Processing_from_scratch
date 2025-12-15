import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import geometry

image_path = "implementation\images\knight_image.jpg"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

second_image = geometry.process_image(grey)

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Original
axes[0].imshow(grey, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")  # Hide axis ticks

# Plot Processed
axes[1].imshow(second_image, cmap="gray")
axes[1].set_title("Processed Image")
axes[1].axis("off")

plt.tight_layout()
plt.show()
