import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2 as cv

# Logic modules
import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform
import logic.text_box_detector.TextBoxDetector as TextBoxDetector

# Py display module
import py_logic.lines_visualizing as py_visual
import py_logic.insure_portrait_orient as portrait
import py_logic.image_as_func_vis as visual
import py_logic.text_analyzer as text_analyzer


# All images

# Lets create expected result for each of this images
# And mark them
# 1 -> Expected good result
# 2 -> Can't say
# 3 -> Expected Bad result

_1_overshadowed_img = "implementation\\images_of_book\\1.jpg"  # 2
_2_exemplary_img = "implementation\\images_of_book\\2.jpg"  # 1
_3_tilted_to_the_side = "implementation\\images_of_book\\3.jpg"  # 3
_4_just_not_good = "implementation\\images_of_book\\4.jpg"  # 3
_5_bad_rotation = "implementation\\images_of_book\\5.jpg"  # TODO: if time will left, handle bad rotation cases
_6_problemetic_on_stitch = "implementation\\images_of_book\\6.jpg"  # 3
_7_overshadowed_banded = "implementation\\images_of_book\\7.jpg"  # 2


image_path = _2_exemplary_img
image = np.array(Image.open(image_path).convert("L"))
grey = np.array(image, dtype=np.float64) / 255.0

start_time = time.time()

canny = EdgeDetector.CannyEdgeDetector()
canny_result = canny.get_canny_img(grey, sigma=1.0, hight_threshold=0.35)
end_time_1 = time.time()

print(f"Time to find edges on first image: {end_time_1-start_time}")


# HOUGH TRANSFORM

rho = 9
theta = 0.261 / 10
threshold = 2000


# Hough transfom
hough_transform = HoughTransform.HoughTransform(canny_result, theta, rho)
my_rotated_image = hough_transform.deskew_image(image, threshold, -np.pi, np.pi)
final_edges = canny.get_canny_img(my_rotated_image, sigma=1.0, hight_threshold=0.24)

end_time_2 = time.time()
print(f"Time to find edges on transformed image: {end_time_2-start_time}")

# Text box detection
final_edges = HoughTransform.conditional_rotation(final_edges)
text_box_detecor = TextBoxDetector.TextBoxDetector(final_edges)

# DEBUG ZONE
print("=========================")
row_signal = text_box_detecor.smooth_row_function()
extream_points = text_box_detecor.find_extream_points()
start_text_rows = text_box_detecor.get_text_rows()
column_function, extream_points_2 = text_box_detecor.seperate_main_text()
clean_text = text_box_detecor.get_clean_text_rows()
text_box_detecor.remove_rows_without_text()

text_rows = text_box_detecor.detect_symbol_boxes(pixel_threshold=1)
print("=========================")

# VISUALIZATION HOUGH TRANSFORM PART
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

# ax[0].imshow(HoughTransform.conditional_rotation(image), cmap="gray")
# ax[0].set_title("Original Image")
# ax[0].axis("off")

# ax[1].imshow(HoughTransform.conditional_rotation(my_rotated_image), cmap="gray")
# ax[1].set_title(f"Deskewed Image")
# ax[1].axis("off")

# ax[2].imshow(HoughTransform.conditional_rotation(final_edges), cmap="gray")
# ax[2].set_title(f"Canny final res")
# ax[2].axis("off")


# VISUALIZATION OF TEXT DETECTOR

row_profile, median = text_analyzer.analyze_text_rows(
    HoughTransform.conditional_rotation(final_edges),
    row_signal,
    extream_points,
    show=False,
)

text_analyzer.analyze_text_columns(
    [el.text_matrix for el in text_rows],
    15,
    second_figure=True,
    col_signals_list=[el._1d_function for el in text_rows],
    zero_sep_points_list=[el.zero_sep_points for el in text_rows],
    potential_sep_points_list=[el.potetional_zero_sep_points for el in text_rows],
)

plt.show()
