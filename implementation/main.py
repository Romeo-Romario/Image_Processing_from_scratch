import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2 as cv

# Logic modules
import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform

# Py display module
import py_logic.lines_visualizing as py_visual
import py_logic.insure_portrait_orient as portrait
import py_logic.image_as_func_vis as visual

image_path = "implementation\\images\\1_1.jpg"
image = np.array(Image.open(image_path).convert("L"))
grey = np.array(image, dtype=np.float64) / 255.0

start_time = time.time()

canny = EdgeDetector.CannyEdgeDetector()
canny_result = canny.get_canny_img(grey, sigma=1.0, hight_threshold=0.25)
end_time_1 = time.time()

print(f"Time to find edges on first image: {end_time_1-start_time}")


# HOUGH TRANSFORM

rho = 9
theta = 0.261 / 10
threshold = 2000


# Hough transfom
# TODO: Should be generalized in one and only global method
hough_transform = HoughTransform.HoughTransform(canny_result, theta, rho)
angle = hough_transform.get_deskew_angle(threshold, -np.pi, np.pi)
print(f"Deskew Angle: {angle}")
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
print("Python Center", center)
M = hough_transform.get_rotation_matrix(center, angle, 1.0)
print(M)

my_rotated_image = hough_transform.deskew(image, M)


final_edges = canny.get_canny_img(my_rotated_image, sigma=1.0, hight_threshold=0.15)

end_time_2 = time.time()
print(f"Time to find edges on transformed image: {end_time_2-start_time}")
# CV part
# =================================
# rotated_image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC)
# rotated_canny_image = cv.warpAffine(canny_result, M, (w, h), flags=cv.INTER_CUBIC)
# =================================

# Plotting
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

# ax[0].imshow(portrait.conditional_rotate(image), cmap="gray")
# ax[0].set_title("Original Image")
# ax[0].axis("off")

# ax[1].imshow(portrait.conditional_rotate(rotated_image), cmap="gray")
# ax[1].set_title(f"Deskewed Image (Angle: {angle:.2f}°)")
# ax[1].axis("off")

# ax[2].imshow(portrait.conditional_rotate(my_rotated_image), cmap="gray")
# ax[2].set_title(f"MY Deskewed Image (Angle: {angle:.2f}°)")
# ax[2].axis("off")

# plt.figure(1)

# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

# ax[0].imshow(portrait.conditional_rotate(canny_result), cmap="gray")
# ax[0].set_title("Canny")
# ax[0].axis("off")

# ax[1].imshow(portrait.conditional_rotate(rotated_canny_image), cmap="gray")
# ax[1].set_title(f"Deskewed Canny")
# ax[1].axis("off")

# ax[2].imshow(portrait.conditional_rotate(my_rotated_image_canny), cmap="gray")
# ax[2].set_title(f"MY Deskewed Canny")
# ax[2].axis("off")


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

ax[0].imshow(portrait.conditional_rotate(image), cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(portrait.conditional_rotate(my_rotated_image), cmap="gray")
ax[1].set_title(f"Deskewed Image (Angle: {angle:.2f}°)")
ax[1].axis("off")

ax[2].imshow(portrait.conditional_rotate(final_edges), cmap="gray")
ax[2].set_title(f"Canny final res {angle:.2f}")
ax[2].axis("off")

plt.show()
