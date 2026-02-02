import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import math
import sys
import os

# 1. Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (go up one level: "../")
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import 'logic' because it sits inside 'project'
import logic.edge_detection.EdgeDetector as EdgeDetector

# --- YOUR MODULE IMPORTS ---
import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform
import py_logic.lines_visualizing as py_visual


# images path
_1 = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images\\square.jpg"
_2 = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images\\tape.jpg"
_3 = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images\\1_1.jpg"
_4 = "D:\\Source\\Diplom\\tryouts\\tryout2_image_deskweing\\implementation\\images\\road.png"

# 1. SETUP & DATA LOADING
image_path = _4
pil_image = Image.open(image_path).convert("L")
# Keep original for display, normalize for Canny
image_arr = np.array(pil_image)
grey_normalized = image_arr.astype(np.float64) / 255.0

# 2. FIGURE 1: STATIC ORIGINAL IMAGE
plt.figure("Original Image", figsize=(6, 6))
plt.imshow(image_arr, cmap="gray")
plt.title("Starting Image (Reference)")
plt.axis("off")

# 3. FIGURE 2: INTERACTIVE DASHBOARD
fig = plt.figure("Interactive Tuner", figsize=(14, 10))
# Create layout: 2 images side-by-side at top, sliders at bottom
ax_canny = fig.add_axes([0.05, 0.45, 0.4, 0.5])  # [left, bottom, width, height]
ax_hough = fig.add_axes([0.55, 0.45, 0.4, 0.5])

# Initial Placeholders
canny_display = ax_canny.imshow(image_arr, cmap="gray")
ax_canny.set_title("Canny Edge Detection")
ax_canny.axis("off")

hough_display = ax_hough.imshow(image_arr, cmap="gray")
ax_hough.set_title("Hough Transform Lines")
ax_hough.axis("off")

# 4. SLIDER SETUP
# Define axes for sliders [left, bottom, width, height]
ax_sigma = fig.add_axes([0.15, 0.30, 0.65, 0.03])
ax_low = fig.add_axes([0.15, 0.25, 0.65, 0.03])
ax_high = fig.add_axes([0.15, 0.20, 0.65, 0.03])
ax_rho = fig.add_axes([0.15, 0.15, 0.65, 0.03])
ax_theta = fig.add_axes([0.15, 0.10, 0.65, 0.03])
ax_thresh = fig.add_axes([0.15, 0.05, 0.65, 0.03])

# Create Sliders
s_sigma = Slider(ax_sigma, "Canny Sigma", 0.1, 10.0, valinit=1.0, valstep=0.1)
s_low = Slider(ax_low, "Low Thresh", 0.0, 1.0, valinit=0.16, valstep=0.01)
s_high = Slider(ax_high, "High Thresh", 0.0, 1.0, valinit=0.58, valstep=0.01)
s_rho = Slider(ax_rho, "Hough Rho", 1.0, 20.0, valinit=9.0, valstep=1.0)
s_theta = Slider(
    ax_theta, "Hough Theta (deg)", 0.1, 5.0, valinit=1.5, valstep=0.1
)  # Slider in Degrees for easier use
s_thresh = Slider(ax_thresh, "Hough Thresh", 10, 3000, valinit=2000, valstep=10)


# 5. UPDATE FUNCTION
def update(val):
    # A. Run Canny (Re-instantiate or just call method depending on your C++ binding)
    # Assuming CannyEdgeDetector is a class we instantiate once, or statics
    canny_detector = EdgeDetector.CannyEdgeDetector()

    # Get values from sliders
    sig = s_sigma.val
    low = s_low.val
    high = s_high.val

    # Run Canny
    # Note: Ensure your binding accepts all these arguments.
    # If not, update bindings.cpp to expose them!
    edges = canny_detector.get_canny_img(
        grey_normalized, sigma=sig, low_threshold=low, hight_threshold=high
    )

    # Update Canny Display
    canny_display.set_data(edges)

    # B. Run Hough Transform
    r_val = s_rho.val
    t_val = np.deg2rad(s_theta.val)  # Convert degrees to radians for C++
    h_thresh = s_thresh.val

    # Instantiate Hough with NEW edges
    ht = HoughTransform.HoughTransform(edges, t_val, r_val)

    # Run Calculation
    # Note: Your C++ returns {accumulator, lines} based on previous context
    # If it only returns lines, remove the 'acc, ' part.
    result = ht.hough_lines(h_thresh, -np.pi / 2, np.pi / 2)

    # Handle return type flexibility (if you changed C++ to return pair or just lines)
    if isinstance(result, tuple) or isinstance(result, list):
        lines = np.array(result[1])  # Assuming [accumulator, lines]
    else:
        lines = np.array(result)

    # C. Visualize
    if lines.size > 0:
        # Pass color image copy to draw lines
        line_image_res, _ = py_visual.draw_lines(image_arr, lines)
        hough_display.set_data(line_image_res)
    else:
        hough_display.set_data(image_arr)  # Show original if no lines found

    fig.canvas.draw_idle()


# 6. REGISTER UPDATES
s_sigma.on_changed(update)
s_low.on_changed(update)
s_high.on_changed(update)
s_rho.on_changed(update)
s_theta.on_changed(update)
s_thresh.on_changed(update)

# Initial run
update(None)

plt.show()
