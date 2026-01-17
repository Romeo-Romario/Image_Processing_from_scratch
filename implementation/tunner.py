import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import logic.edge_detection.EdgeDetector as EdgeDetector

# 1. Load Data
image_path = "implementation\\images\\road.png"
try:
    image = Image.open(image_path).convert("L")
    grey = np.array(image, dtype=np.float64) / 255.0
except FileNotFoundError:
    print(f"Error: Could not find image at {image_path}")
    exit()

# 2. Initialize C++ Detector
# We create the instance ONCE to reuse memory buffers (faster updates)
canny = EdgeDetector.CannyEdgeDetector()

# 3. Setup the Plot
# We need space at the bottom for sliders, so we adjust 'bottom=0.25'
fig, (ax_orig, ax_edge) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Initial calculation
initial_sigma = 1.0
initial_high = 0.4
initial_low = 0.06
result = canny.get_canny_img(
    grey, sigma=initial_sigma, hight_threshold=initial_high, low_threshold=initial_low
)

# Display images
ax_orig.imshow(grey, cmap="gray")
ax_orig.set_title("Original")
ax_orig.axis("off")

# We save the plot object 'img_plot' to update its data later (faster than re-plotting)
img_plot = ax_edge.imshow(result, cmap="gray")
ax_edge.set_title("Canny Result")
ax_edge.axis("off")

# 4. Create Sliders
# Define axes positions [left, bottom, width, height]
ax_sigma = plt.axes([0.25, 0.15, 0.50, 0.03])
ax_high = plt.axes([0.25, 0.10, 0.50, 0.03])
ax_low = plt.axes([0.25, 0.05, 0.50, 0.03])

s_sigma = Slider(ax_sigma, "Sigma", 0.1, 5.0, valinit=initial_sigma, valstep=0.1)
s_high = Slider(ax_high, "High Thresh", 0.0, 1.0, valinit=initial_high, valstep=0.01)
s_low = Slider(ax_low, "Low Thresh", 0.0, 1.0, valinit=initial_low, valstep=0.01)


# 5. The Update Function
def update(val):
    # Get current values from sliders
    sig = s_sigma.val
    h_th = s_high.val
    l_th = s_low.val

    # Enforce logic: High must be >= Low
    if l_th > h_th:
        # Optional: Force low to be lower, or just let it produce weird results
        pass

    # Call C++ (Fast!)
    # Note: Pass 'grey' every time so the wrapper re-calculates on the original data
    new_edges = canny.get_canny_img(
        grey, sigma=sig, hight_threshold=h_th, low_threshold=l_th
    )

    # Update the image data directly
    img_plot.set_data(new_edges)

    # Redraw the figure
    fig.canvas.draw_idle()


# Connect sliders to function
s_sigma.on_changed(update)
s_high.on_changed(update)
s_low.on_changed(update)

plt.show()
