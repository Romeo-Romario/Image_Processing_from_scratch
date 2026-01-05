import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import logic.edge_detection.EdgeDetector as EdgeDetector

image_path = ".\implementation\images\zebra2.png"
image = Image.open(image_path).convert("L")
grey = np.array(image, dtype=np.float64) / 255.0

start_time = time.time()

canny = EdgeDetector.CannyEdgeDetector()

canny_result = canny.get_canny_img(grey, hight_threshold=0.4)

end_time = time.time()

print(f"Time to find edges: {end_time-start_time}")

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
ax[0].imshow(grey, cmap="gray")
ax[1].imshow(canny_result, cmap="gray")
ax[0].set_title("Beggining image")
ax[1].set_title("Canny detector")

plt.show()
