import numpy as np
import matplotlib.pyplot as plt


def analyze_text_rows(binary_img):
    """
    Computes and visualizes the Horizontal Projection Profile.
    """
    # 1. Calculate Projection
    row_signal = np.sum(binary_img, axis=1)

    # 2. Calculate Median
    non_zero_signal = row_signal[row_signal > 0]
    median_val = np.median(non_zero_signal) if len(non_zero_signal) > 0 else 0

    # 3. Visualization
    height, width = binary_img.shape
    y_coords = np.arange(height)

    # REMOVED sharey=True to prevent aspect ratio fighting
    plt.figure(1)
    fig, (ax_img, ax_plot) = plt.subplots(
        1, 2, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 2]}
    )

    # --- LEFT: The Image ---
    ax_img.imshow(binary_img, cmap="gray", aspect="auto")
    # NOTE: aspect='auto' fills the plot box.
    # If you want TRUE natural proportions, use aspect='equal' or remove the arg.
    # But aspect='auto' ensures it lines up perfectly with the graph on the right.

    ax_img.set_title("Deskewed Edge Map")
    ax_img.set_ylabel("Row Number (Y)")

    # --- RIGHT: The 1D Function ---
    ax_plot.plot(row_signal, y_coords, color="black", label="Row Signal")
    ax_plot.fill_betweenx(y_coords, 0, row_signal, alpha=0.3, color="gray")

    ax_plot.axvline(
        x=median_val, color="red", linestyle="--", label=f"Median: {median_val:.1f}"
    )

    # --- SYNCING AXES MANUALLY ---
    # Invert Y on the plot to match the image (0 at top)
    ax_plot.invert_yaxis()

    # Force both to have the same Y limits (0 to height)
    ax_plot.set_ylim(height, 0)
    ax_img.set_ylim(height, 0)

    # Styling
    ax_plot.set_title("Horizontal Projection Profile")
    ax_plot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_plot.legend()

    return row_signal, median_val
