import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import geometry

image_path = "implementation\images\zebra.jpg"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

edge_detector = geometry.EdgeDetector(grey)

edge_detector.convolve_image(1.0, True)

images = edge_detector.get_image_gradients()

print(images)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(grey, cmap="gray")
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(images[0], cmap="gray")
axes[0, 1].set_title("Derivative X (Vertical Edges)")

axes[1, 0].imshow(images[1], cmap="gray")
axes[1, 0].set_title("Derivative Y (Horizontal Edges)")

axes[1, 1].imshow(images[2], cmap="gray")
axes[1, 1].set_title("Second Derivative / Magnitude")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.figure(2)

print("one")
orientation_images = edge_detector.get_image_gradient_orientation()
print(orientation_images)
plt.imshow(orientation_images[0], cmap="gray")
plt.show()
