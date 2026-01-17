import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2 as cv

import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform
import py_logic.lines_visualizing as py_visual

image_path = "implementation\\images\\road.png"
image = Image.open(image_path).convert("L")
image = np.array(image)
grey = np.array(image, dtype=np.float64) / 255.0

start_time = time.time()

canny = EdgeDetector.CannyEdgeDetector()

canny_result = canny.get_canny_img(
    grey,
    sigma=1,
)

end_time = time.time()

print(f"Time to find edges: {end_time-start_time}")


# HOUGH TRANSFORM

rho = 9
theta = 0.261
threshold = 750


hough_transform = HoughTransform.HoughTransform(canny_result, theta, rho)

accumulator, lines = hough_transform.hough_lines(threshold, -np.pi / 2, np.pi / 2)
lines = np.array(lines)
accumulator = np.array(accumulator)


# Show the image with the lines found
lines_img, mask = py_visual.draw_lines(image, lines)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
ax[0].imshow(grey, cmap="gray")
ax[1].imshow(canny_result, cmap="gray")
ax[0].set_title("Beggining image")
ax[1].set_title("Canny detector")

plt.figure(2)
plt.imshow(accumulator, cmap="gray")
plt.title("Parameter space")
plt.show()


win_name = "hough"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
cv.imshow(win_name, lines_img)
cv.waitKey(0)
cv.destroyAllWindows()
