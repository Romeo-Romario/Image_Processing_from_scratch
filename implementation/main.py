import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import logic.geometry as geometry

image_path = "implementation\images\image.png"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

start_time = time.time()
edge_detector = geometry.EdgeDetector(grey)

edge_detector.convolve_image(0.5, False)

images = edge_detector.get_image_gradients()

grad_orientation_data = edge_detector.get_image_gradient_orientation()

grad_non_max_supress = edge_detector.get_non_max_suppresion()

thresholded_img = edge_detector.get_thresholded_img(
    np.max(grad_non_max_supress),
    np.min(grad_non_max_supress),
    np.mean(grad_non_max_supress),
)

hyst_img = edge_detector.get_hysteresis_img()
end_time = time.time()

print(f"Time to find edges: {end_time-start_time}")

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
ax[0].imshow(images[2], cmap="gray")
ax[1].imshow(grad_non_max_supress, cmap="gray")
ax[0].set_title("Gradient Orientation")
ax[1].set_title("Non Maxima suppresion")

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
ax[0].imshow(thresholded_img, cmap="gray")
ax[1].imshow(hyst_img, cmap="gray")
ax[0].set_title("Tresholded")
ax[1].set_title("Hysteresis")

plt.show()
