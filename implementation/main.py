import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2 as cv

import torch
import os
import sys


# Logic modules
import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform
import logic.text_box_detector.TextBoxDetector as TextBoxDetector

# Py display module
import py_logic.lines_visualizing as py_visual
import py_logic.insure_portrait_orient as portrait
import py_logic.image_as_func_vis as visual
import py_logic.text_analyzer as text_analyzer

current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Point directly to the 'machine_learning' subfolder
ml_dir = os.path.join(current_dir, "machine_learning")
sys.path.append(ml_dir)
from machine_learning.custom_cnn import UkrainianOCRResNet

# ==========================================
# ML MODEL INITIALIZATION
# ==========================================
print("Waking up the neural network...")

# Define paths to your best run
weights_path = os.path.join(ml_dir, r"runs\exp_004\best_model.pth")
mapping_path = os.path.join(ml_dir, r"runs\exp_004\class_mapping.txt")

# 1. Load the dictionary so the model knows how many output nodes to create
ocr_class_mapping = {}
with open(mapping_path, "r", encoding="utf-8") as f:
    for line in f:
        idx, char = line.strip().split(":")
        ocr_class_mapping[int(idx)] = char

num_classes = len(ocr_class_mapping)

# 2. Choose the hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Build the ResNet architecture and pour the trained weights into it
ocr_model = UkrainianOCRResNet(num_classes=num_classes)
ocr_model.load_state_dict(
    torch.load(weights_path, map_location=device, weights_only=True)
)
ocr_model.to(device)

# 4. CRITICAL: Lock the model into Evaluation Mode
ocr_model.eval()

print(f"Model successfully loaded with {num_classes} classes on {device}!")
# ==========================================

_1_ml = r"implementation\images\book_images\IMG_20260320_002833.jpg"
_2_ml = r"implementation\images\book_images\IMG_20260320_002847.jpg"
_3_ml = r"implementation\images\book_images\IMG_20260320_002847.jpg"
_4_ml = r"implementation\images\book_images\IMG_20260320_113958.jpg"  # Absolute cinema
_4_1_ml = (
    r"implementation\images\book_images\IMG_20260321_171615.jpg"  # Very good example
)


image_path = _4_ml
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

# Time your custom C++ detection
start_custom_boxes = time.time()
text_rows = text_box_detecor.detect_symbol_boxes(
    density_threshold=6.2, pixel_threshold=1
)
custom_time = time.time() - start_custom_boxes
print(f"Time to extract symbol boxes (Custom C++): {custom_time:.3f} seconds")


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
