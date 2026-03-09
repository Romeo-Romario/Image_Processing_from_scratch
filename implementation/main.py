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

_1_overshadowed_img = "implementation\\images_of_book\\1.jpg"  # 2 -> 3
_2_exemplary_img = "implementation\\images_of_book\\2.jpg"  # 1 -> 1
_3_tilted_to_the_side = "implementation\\images_of_book\\3.jpg"  # 3 -> 3
_4_just_not_good = "implementation\\images_of_book\\4.jpg"  # 3 -> 3
_5_problemetic_on_stitch = "implementation\\images_of_book\\6.jpg"  # 3 -> 3
_6_overshadowed_banded = "implementation\\images_of_book\\7.jpg"  # 2


_1_ml = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images_ml\\IMG_20260225_194826.jpg"
_2_ml = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images_ml\\IMG_20260225_194905.jpg"
_3_ml = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images_ml\\IMG_20260225_201114.jpg"
_4_ml = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images_ml\\IMG_20260225_200958.jpg"
_4_1_ml = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images_ml\\IMG_20260225_200958_1.jpg"


image_path = _4_1_ml
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
final_edges = canny.get_canny_img(my_rotated_image, sigma=1.0, hight_threshold=0.20)

end_time_2 = time.time()
print(f"Time to find edges on transformed image: {end_time_2-start_time}")

# Text box detection
final_edges = HoughTransform.conditional_rotation(final_edges)
text_box_detecor = TextBoxDetector.TextBoxDetector(final_edges)


# DEBUG ZONE
print("=========================")
start_custom_boxes = time.time()
# Time your custom C++ detection
text_rows = text_box_detecor.detect_symbol_boxes(
    density_threshold=6.5, pixel_threshold=1
)
custom_time = time.time() - start_custom_boxes

print(f"Time to extract symbol boxes (Custom C++): {custom_time:.3f} seconds")
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

# row_profile, median = text_analyzer.analyze_text_rows(
#     HoughTransform.conditional_rotation(final_edges),
#     row_signal,
#     extream_points,
#     show=True,
# )

# text_analyzer.analyze_text_columns(
#     [el.text_matrix for el in text_rows],
#     15,
#     second_figure=False,
#     col_signals_list=[el._1d_function for el in text_rows],
#     zero_sep_points_list=[el.zero_sep_points for el in text_rows],
#     potential_sep_points_list=[el.potetional_zero_sep_points for el in text_rows],
# )


# Use the new comparison function
# Pass the original deskewed image, the extracted rows, and let it plot
text_analyzer.compare_symbol_boxes(
    HoughTransform.conditional_rotation(my_rotated_image),
    text_rows,
    custom_time=custom_time,
    show=True,
)


plt.show()
