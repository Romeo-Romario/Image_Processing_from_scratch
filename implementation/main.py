import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import geometry

image_path = "implementation\images\image.png"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

edge_detector = geometry.EdgeDetector(grey)

edge_detector.convolve_image(1.0, True)

images = edge_detector.generate_matrixes()


# Create a figure with a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Plot Original Image (Top Left)
axes[0, 0].imshow(grey, cmap="gray")
axes[0, 0].set_title("Original Image")

# 2. Plot dI/dX (Top Right)
# Based on your C++ order: matrix_ptrs = {&dI_dX, &dI_dY, &dI2}
axes[0, 1].imshow(images[0], cmap="gray")
axes[0, 1].set_title("Derivative X (Vertical Edges)")

# 3. Plot dI/dY (Bottom Left)
axes[1, 0].imshow(images[1], cmap="gray")
axes[1, 0].set_title("Derivative Y (Horizontal Edges)")

print("Max and min values", images[2].min(), images[2].max())
# 4. Plot dI2 (Bottom Right)
axes[1, 1].imshow(images[2], cmap="gray")
axes[1, 1].set_title("Second Derivative / Magnitude")

# Hide axis ticks for all plots
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
