import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_3d_region(ax, img, title, start_row, start_col, size=20):
    """
    Plots a specific small region (size x size) of the image as a 3D surface.
    """
    # 1. Define the Crop (Slice)
    end_row = start_row + size
    end_col = start_col + size

    # Safety check to ensure we don't go out of bounds
    max_rows, max_cols = img.shape
    if end_row > max_rows or end_col > max_cols:
        print(f"Warning: Region {end_row}x{end_col} is out of bounds. Clipping.")
        end_row = min(end_row, max_rows)
        end_col = min(end_col, max_cols)

    # 2. Slice the Data (The "Zoom")
    # We take the data ONLY from the region we want
    region_img = img[start_row:end_row, start_col:end_col]

    # 3. Create Grid corresponding to that specific region
    # We use start_col/start_row so the axis labels show the REAL pixel coordinates
    x = np.arange(start_col, end_col, 1)
    y = np.arange(start_row, end_row, 1)
    X, Y = np.meshgrid(x, y)

    # 4. Plot (No stride needed because 20x20 is very small and fast)
    surf = ax.plot_surface(
        X,
        Y,
        region_img,
        cmap=cm.viridis,
        linewidth=0.5,  # Small line width helps visualize the "grid" of pixels
        edgecolor="k",  # Black edges to see individual pixels clearly
        antialiased=True,
    )

    ax.set_title(
        f"{title}\nRegion: ({start_col}, {start_row}) to ({end_col}, {end_row})"
    )
    ax.set_xlabel("X (Columns)")
    ax.set_ylabel("Y (Rows)")
    ax.set_zlabel("Intensity")

    return surf


# CODE TO PLOT IN main.py

# fig = plt.figure(figsize=(16, 8))

# # Define where to look (e.g., center of the image)
# center_y = grey.shape[0] // 2
# center_x = grey.shape[1] // 2

# # Plot 1: Original Grey Image Region
# ax1 = fig.add_subplot(1, 2, 1, projection="3d")
# visual.plot_3d_region(
#     ax1,
#     grey,
#     "Original Grey",
#     start_row=center_y + 20,
#     start_col=center_x + 20,
#     size=20,
# )

# # Plot 2: Convolved Image Region (Same coordinates)
# ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# visual.plot_3d_region(
#     ax2,
#     convolved_img,
#     "Convolved (Edges)",
#     start_row=center_y + 20,
#     start_col=center_x + 20,
#     size=20,
# )
