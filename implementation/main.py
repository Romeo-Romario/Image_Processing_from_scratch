import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import logic.geometry as geometry

image_path = "implementation\\images\\zebra2.png"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

edge_detector = geometry.EdgeDetector(grey)

edge_detector.convolve_image(1.0, False)

images = edge_detector.get_image_gradients()

# print(images)
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# axes[0, 0].imshow(grey, cmap="gray")
# axes[0, 0].set_title("Original Image")

# axes[0, 1].imshow(images[0], cmap="gray")
# axes[0, 1].set_title("Derivative X (Vertical Edges)")

# axes[1, 0].imshow(images[1], cmap="gray")
# axes[1, 0].set_title("Derivative Y (Horizontal Edges)")

# axes[1, 1].imshow(images[2], cmap="gray")
# axes[1, 1].set_title("Second Derivative / Magnitude")

# for ax in axes.flat:
#     ax.axis("off")

grad_orientation_data = edge_detector.get_image_gradient_orientation()

grad_non_max_supress = edge_detector.get_non_max_suppresion()

thresholded_img = edge_detector.get_thresholded_img(
    np.max(grad_non_max_supress),
    np.min(grad_non_max_supress),
    np.mean(grad_non_max_supress),
)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
ax[0].imshow(images[2], cmap="gray")
ax[1].imshow(grad_non_max_supress, cmap="gray")
ax[2].imshow(thresholded_img, cmap="gray")
ax[0].set_title("Gradient Orientation")
ax[1].set_title("Non Maxima suppresion")
ax[2].set_title("Tresholded")

# fig, ax = plt.subplots(1, 2, figsize=(15, 15))
# ax[0].imshow(images[2], cmap="gray")
# ax[1].imshow(grad_non_max_supress, cmap="gray")
# ax[0].set_title("Gradient Orientation")
# ax[1].set_title("Non Maxima suppresion")

plt.show()
