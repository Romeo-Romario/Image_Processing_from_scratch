import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def analyze_text_rows(
    binary_img: np.array,
    row_signal: np.array,
    indexes_of_extream_points: Optional[List[bool]] = None,
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
    plt.show()

    return row_signal, median_val
