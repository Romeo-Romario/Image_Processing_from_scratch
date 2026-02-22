import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import matplotlib.patches as patches

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import time

# To compare with actual engine
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "D:\\Libaries\\Tesseract\\tesseract.exe"


def analyze_text_rows(
    binary_img: np.array,
    row_signal: np.array,
    indexes_of_extream_points: Optional[List[bool]] = None,
    show: bool = True,
):
    """
    Computes and visualizes the Horizontal Projection Profile.

    indexes_of_extream_points: Optional boolean array (True/False) matching image height.
                               Rows marked True will be highlighted with a horizontal line.
    """
    row_signal = np.array(row_signal)

    # 2. Calculate Median
    non_zero_signal = row_signal[row_signal > 0]
    median_val = np.median(non_zero_signal) if len(non_zero_signal) > 0 else 0

    # 3. Visualization
    height, width = binary_img.shape
    y_coords = np.arange(height)

    # Debug print (preserved from your snippet)
    print(height, width, len(y_coords), len(row_signal))

    # Create Plot
    fig, (ax_img, ax_plot) = plt.subplots(
        1, 2, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 2]}
    )

    # --- LEFT: The Image ---
    ax_img.imshow(binary_img, cmap="gray", aspect="auto")
    ax_img.set_title("Deskewed Edge Map")
    ax_img.set_ylabel("Row Number (Y)")

    # --- RIGHT: The 1D Function ---
    ax_plot.plot(row_signal, y_coords, color="black", label="Row Signal")
    ax_plot.fill_betweenx(y_coords, 0, row_signal, alpha=0.3, color="gray")

    # Draw Median Line
    ax_plot.axvline(
        x=median_val, color="red", linestyle="--", label=f"Median: {median_val:.1f}"
    )

    # --- NEW: VISUALIZE EXTREME POINTS ---
    if indexes_of_extream_points is not None:
        # Convert boolean mask [True, False, True...] to indices [0, 2, ...]
        # We assume the list is the same length as the image height
        target_indices = np.where(indexes_of_extream_points)[0]

        if len(target_indices) > 0:
            # 1. Draw lines on the Image (Left)
            # hlines(y, xmin, xmax)
            ax_img.hlines(
                target_indices,
                0,
                width,
                colors="lime",
                linestyles="solid",
                linewidth=1,
                alpha=0.7,
            )

            # 2. Draw lines on the Plot (Right)
            # We determine the max signal value just to set the line length correctly
            max_sig = np.max(row_signal) if len(row_signal) > 0 else 1

            ax_plot.hlines(
                target_indices,
                0,
                max_sig,
                colors="lime",
                linestyles="--",
                linewidth=1,
                alpha=0.7,
                label="Extrema",
            )

    # --- SYNCING AXES MANUALLY ---
    ax_plot.invert_yaxis()
    ax_plot.set_ylim(height, 0)
    ax_img.set_ylim(height, 0)

    # --- Styling ---
    ax_plot.set_title("Horizontal Projection Profile")
    ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_plot.legend()

    plt.tight_layout()
    if show:
        plt.show()

    return row_signal, median_val


def analyze_text_columns(
    text_rows_list: List[np.array],
    row_index: int,
    col_signals_list: Optional[List[np.array]] = None,
    zero_sep_points_list: Optional[List[List[int]]] = None,
    potential_sep_points_list: Optional[List[List[int]]] = None,
    pixel_threshold: float = 0.0,
    second_figure: bool = False,
):

    if row_index < 0 or row_index >= len(text_rows_list):
        print(
            f"Error: row_index {row_index} is out of bounds (Size: {len(text_rows_list)})"
        )
        return None, 0

    binary_img = np.array(text_rows_list[row_index])
    height, width = binary_img.shape
    x_coords = np.arange(width)

    col_signal = None
    if col_signals_list is not None:
        if row_index < len(col_signals_list):
            col_signal = np.array(col_signals_list[row_index])
        else:
            print(
                f"Warning: col_signals_list provided but row_index {row_index} is out of bounds."
            )

    if col_signal is None:
        col_signal = np.sum(binary_img, axis=0)

    current_zero_points = None
    if zero_sep_points_list is not None and row_index < len(zero_sep_points_list):
        current_zero_points = zero_sep_points_list[row_index]

    current_potential_points = None
    if potential_sep_points_list is not None and row_index < len(
        potential_sep_points_list
    ):
        current_potential_points = potential_sep_points_list[row_index]

    non_zero_signal = col_signal[col_signal > 0]
    median_val = np.median(non_zero_signal) if len(non_zero_signal) > 0 else 0
    division_threshold = pixel_threshold * 256.0

    if second_figure:
        plt.figure(1)

    fig, (ax_img, ax_plot) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    ax_img.imshow(binary_img, cmap="gray", aspect="auto")
    ax_img.set_title(f"Text Row #{row_index} (Image)")
    ax_img.set_ylabel("Row Height")

    ax_plot.plot(x_coords, col_signal, color="black", label="Column Signal")
    ax_plot.fill_between(x_coords, 0, col_signal, alpha=0.3, color="gray")

    ax_plot.axhline(
        y=median_val, color="red", linestyle="--", label=f"Median: {median_val:.1f}"
    )

    if pixel_threshold > 0:
        ax_plot.axhline(
            y=division_threshold,
            color="blue",
            linestyle=":",
            label=f"Threshold: {division_threshold:.1f}",
        )

    max_sig = np.max(col_signal) if len(col_signal) > 0 else 1

    if current_zero_points is not None and len(current_zero_points) > 0:
        ax_img.vlines(
            current_zero_points,
            0,
            height,
            colors="lime",
            linestyles="solid",
            linewidth=1,
            alpha=0.7,
        )
        ax_plot.vlines(
            current_zero_points,
            0,
            max_sig,
            colors="lime",
            linestyles="--",
            linewidth=1.5,
            alpha=0.8,
            label="Zero Cuts",
        )

    if current_potential_points is not None and len(current_potential_points) > 0:
        ax_img.vlines(
            current_potential_points,
            0,
            height,
            colors="orange",
            linestyles="solid",
            linewidth=1,
            alpha=0.7,
        )
        ax_plot.vlines(
            current_potential_points,
            0,
            max_sig,
            colors="orange",
            linestyles="--",
            linewidth=1.5,
            alpha=0.8,
            label="Potential Cuts",
        )

    ax_plot.set_title("Vertical Projection Profile")
    ax_plot.set_xlabel("Column Index (X)")
    ax_plot.set_ylabel("Pixel Sum")
    ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_plot.legend(loc="upper right")

    ax_img.set_xlim(0, width)

    plt.tight_layout()
    plt.show()

    return col_signal, median_val


def visualize_symbol_boxes(original_image, text_rows, show=True):
    """
    Draws bounding boxes around individual symbols detected by the C++ backend.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(original_image, cmap="gray")
    ax.set_title("Detected Symbol Bounding Boxes")

    for row in text_rows:
        # Each row has a list of symbol limits (pairs of Points)
        for top_left, bottom_right in row.symbols_limits:
            # Assuming pybind11 exposes the C++ Point struct with .x and .y attributes
            x = top_left.x
            y = top_left.y
            width = bottom_right.x - top_left.x
            height = bottom_right.y - top_left.y

            # Create a Rectangle patch (x, y of top-left corner, width, height)
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=1.5, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

    plt.tight_layout()
    if show:
        plt.show()


def visualize_tesseract_boxes(image):
    """
    Runs Tesseract OCR on the image and draws character-level bounding boxes.
    """
    # Tesseract expects a standard uint8 image (0-255)
    if image.dtype == np.float64:
        img_uint8 = (
            (image * 255).astype(np.uint8)
            if np.max(image) <= 1.0
            else image.astype(np.uint8)
        )
    else:
        img_uint8 = image

    h, w = img_uint8.shape

    # Get character bounding boxes
    # Format: character left bottom right top page
    boxes_data = pytesseract.image_to_boxes(img_uint8)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img_uint8, cmap="gray")
    ax.set_title("Tesseract OCR Symbol Bounding Boxes")

    for b in boxes_data.splitlines():
        b = b.split(" ")
        char = b[0]
        # Tesseract returns coordinates from the bottom-left corner
        left, bottom, right, top = int(b[1]), int(b[2]), int(b[3]), int(b[4])

        # Convert to matplotlib's top-left origin system
        x = left
        y = h - top
        box_width = right - left
        box_height = top - bottom

        # Draw a blue rectangle for Tesseract
        rect = patches.Rectangle(
            (x, y),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Optional: Add the predicted character above the box
        # ax.text(x, y - 2, char, color='blue', fontsize=8)

    plt.tight_layout()
    plt.show()


def compare_symbol_boxes(original_image, text_rows, custom_time="", show=True):
    """
    Draws custom algorithm bounding boxes on the left, and Tesseract boxes on the right.
    Calculates and displays the processing time for Tesseract.
    """
    # 1. Prepare Image for Tesseract
    if original_image.dtype == np.float64:
        img_uint8 = (
            (original_image * 255).astype(np.uint8)
            if np.max(original_image) <= 1.0
            else original_image.astype(np.uint8)
        )
    else:
        img_uint8 = original_image

    h, w = img_uint8.shape

    # 2. Run Tesseract and time it
    start_tesseract = time.time()
    boxes_data = pytesseract.image_to_boxes(img_uint8)
    tesseract_time = time.time() - start_tesseract

    # 3. Setup the dual plot
    fig, (ax_custom, ax_tess) = plt.subplots(
        1, 2, figsize=(20, 10), sharex=True, sharey=True
    )

    ax_custom.imshow(original_image, cmap="gray")
    ax_custom.set_title(f"Custom C++ Bounding Boxes (Time: {custom_time:.3f}s)")

    ax_tess.imshow(img_uint8, cmap="gray")
    ax_tess.set_title(f"Tesseract OCR Bounding Boxes (Time: {tesseract_time:.3f}s)")

    # 4. Draw Custom Boxes (Red)
    for row in text_rows:
        for top_left, bottom_right in row.symbols_limits:
            x = top_left.x
            y = top_left.y
            box_w = bottom_right.x - top_left.x
            box_h = bottom_right.y - top_left.y

            rect = patches.Rectangle(
                (x, y), box_w, box_h, linewidth=1.5, edgecolor="red", facecolor="none"
            )
            ax_custom.add_patch(rect)

    # 5. Draw Tesseract Boxes (Blue)
    for b in boxes_data.splitlines():
        b = b.split(" ")
        if len(b) >= 5:
            left, bottom, right, top = int(b[1]), int(b[2]), int(b[3]), int(b[4])

            # Convert to matplotlib's top-left origin system
            x = left
            y = h - top
            box_width = right - left
            box_height = top - bottom

            rect = patches.Rectangle(
                (x, y),
                box_width,
                box_height,
                linewidth=1.5,
                edgecolor="blue",
                facecolor="none",
            )
            ax_tess.add_patch(rect)

    plt.tight_layout()
    if show:
        plt.show()
