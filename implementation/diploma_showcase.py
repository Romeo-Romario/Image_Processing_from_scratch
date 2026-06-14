import os
import sys
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

# --- CUSTOM PIPELINE MODULES ---
import logic.edge_detection.EdgeDetector as EdgeDetector
import logic.hough_transform.HoughTransform as HoughTransform
import logic.text_box_detector.TextBoxDetector as TextBoxDetector
import py_logic.text_analyzer as text_analyzer
import py_logic.lines_visualizing as py_visual

# Set Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"D:\Libaries\Tesseract\tesseract.exe"


def run_benchmarks():
    print("=" * 50)
    print("INITIATING PIPELINE BENCHMARKS")
    print("=" * 50)

    # 1. Load the Image
    image_path = r"implementation\images\book_images\IMG_20260320_113944.jpg"
    image = np.array(Image.open(image_path).convert("L"))

    # Pre-process for different libraries
    img_float = np.array(image, dtype=np.float64) / 255.0  # For custom C++ math
    img_uint8 = image.copy()  # For OpenCV, Tesseract, and plotting

    # ==========================================
    # STAGE 1: EDGE DETECTION (CANNY)
    # ==========================================
    print("\n--- STAGE 1: EDGE DETECTION ---")

    # 1A. OpenCV Canny
    start = time.time()
    opencv_canny = cv.Canny(img_uint8, 50, 150)
    opencv_canny_time = time.time() - start
    print(f"OpenCV Canny Time:         {opencv_canny_time:.4f}s")

    # 1B. Custom C++ Canny
    start = time.time()
    custom_edge_detector = EdgeDetector.CannyEdgeDetector()
    custom_canny = custom_edge_detector.get_canny_img(
        img_float, sigma=1.0, hight_threshold=0.35
    )
    custom_canny_time = time.time() - start
    print(f"Custom C++ Canny Time:     {custom_canny_time:.4f}s")

    # Plot Stage 1
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig1.canvas.manager.set_window_title("Stage 1: Edge Detection Comparison")

    ax1[0].imshow(opencv_canny, cmap="gray")
    ax1[0].set_title(f"OpenCV Canny\nTime: {opencv_canny_time:.4f}s")
    ax1[0].axis("off")

    ax1[1].imshow(HoughTransform.conditional_rotation(custom_canny), cmap="gray")
    ax1[1].set_title(f"Custom C++ Canny\nTime: {custom_canny_time:.4f}s")
    ax1[1].axis("off")

    # ==========================================
    # STAGE 2: HOUGH TRANSFORM (LINE DETECTION)
    # ==========================================
    print("\n--- STAGE 2: HOUGH TRANSFORM ---")

    # 2A. OpenCV Hough Lines
    start = time.time()
    cv_lines = cv.HoughLines(opencv_canny, 1, np.pi / 180, 200)
    opencv_hough_time = time.time() - start
    print(f"OpenCV Hough Time:         {opencv_hough_time:.4f}s")

    # Draw OpenCV Lines (Outside the timer!)
    cv_hough_vis = cv.cvtColor(img_uint8, cv.COLOR_GRAY2BGR)
    if cv_lines is not None:
        for i in range(min(len(cv_lines), 20)):  # Draw top 20 lines
            rho, theta = cv_lines[i][0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
            pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
            cv.line(cv_hough_vis, pt1, pt2, (255, 0, 0), 2, cv.LINE_AA)

    # 2B. Custom C++ Hough Transform & Deskew (Unified Pass)
    start = time.time()
    rho_res, theta_res, threshold = 9, 0.261 / 10, 2000
    custom_hough = HoughTransform.HoughTransform(custom_canny, theta_res, rho_res)

    # Calculate accumulator, find lines, and actively deskew the image in one pass
    my_rotated_image = custom_hough.deskew_image(img_uint8, threshold, -np.pi, np.pi)

    # Extract the populated coordinates
    _, polar_coords = custom_hough.get_accumulator_and_polar_coordinates()
    custom_hough_time = time.time() - start
    print(f"Custom C++ Hough & Deskew: {custom_hough_time:.4f}s")

    # Draw Custom Lines (Outside the timer!)
    custom_hough_vis, _ = py_visual.draw_lines(img_uint8, np.array(polar_coords))

    # Helper function to pass 3D RGB images through your 2D C++ rotation
    def rotate_color_image(img_3d):
        c1, c2, c3 = cv.split(img_3d)
        return cv.merge(
            [
                HoughTransform.conditional_rotation(c1),
                HoughTransform.conditional_rotation(c2),
                HoughTransform.conditional_rotation(c3),
            ]
        )

    # Plot Stage 2
    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig2.canvas.manager.set_window_title("Stage 2: Hough Lines Comparison")

    cv_hough_rgb = cv.cvtColor(cv_hough_vis, cv.COLOR_BGR2RGB)
    ax2[0].imshow(rotate_color_image(cv_hough_rgb))
    ax2[0].set_title(f"OpenCV Hough Lines\nTime: {opencv_hough_time:.4f}s")
    ax2[0].axis("off")

    if len(custom_hough_vis.shape) == 3:
        custom_hough_rgb = cv.cvtColor(
            custom_hough_vis.astype(np.uint8), cv.COLOR_BGR2RGB
        )
        custom_hough_final = rotate_color_image(custom_hough_rgb)
    else:
        custom_hough_final = HoughTransform.conditional_rotation(custom_hough_vis)

    ax2[1].imshow(custom_hough_final)
    ax2[1].set_title(f"Custom C++ Hough Lines\n(Computed in Unified Pass)")
    ax2[1].axis("off")

    # ==========================================
    # STAGE 3: IMAGE DESKEWING (ROTATION)
    # ==========================================
    print("\n--- STAGE 3: IMAGE DESKEWING ---")

    # 3A. OpenCV Deskewing
    start = time.time()
    angles = []
    # Extract the dominant angle from the horizontal text lines
    if cv_lines is not None:
        for line in cv_lines:
            rho, theta = line[0]
            deg = np.degrees(theta)
            if 45 < deg < 135:  # Only look at horizontal-ish lines
                angles.append(deg - 90)

    dominant_angle = np.median(angles) if angles else 0.0

    (h, w) = img_uint8.shape[:2]
    center = (w // 2, h // 2)
    # Build rotation matrix and warp the image
    M = cv.getRotationMatrix2D(center, dominant_angle, 1.0)
    cv_deskewed = cv.warpAffine(
        img_uint8, M, (w, h), flags=cv.INTER_CUBIC, borderValue=255
    )
    cv_deskewed_time = time.time() - start
    print(
        f"OpenCV Deskewing Time:     {cv_deskewed_time:.4f}s (Angle: {dominant_angle:.2f} deg)"
    )

    # 3B. Custom C++ Deskewing
    # The deskewing time is shared with Stage 2 because of the unified C++ pass!
    custom_deskewed_final = HoughTransform.conditional_rotation(my_rotated_image)

    # Plot Stage 3
    fig3, ax3 = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig3.canvas.manager.set_window_title("Stage 3: Image Deskewing Comparison")

    # Pass OpenCV deskewed image through conditional rotation to match orientation
    ax3[0].imshow(HoughTransform.conditional_rotation(cv_deskewed), cmap="gray")
    ax3[0].set_title(f"OpenCV Deskewed Image\nTime: {cv_deskewed_time:.4f}s")
    ax3[0].axis("off")

    ax3[1].imshow(custom_deskewed_final, cmap="gray")
    ax3[1].set_title(
        f"Custom C++ Deskewed Image\n(Unified Pass Time: {custom_hough_time:.4f}s)"
    )
    ax3[1].axis("off")

    # ==========================================
    # STAGE 4: SYMBOL SEGMENTATION (BOUNDING BOXES)
    # ==========================================
    print("\n--- STAGE 4: BOUNDING BOX EXTRACTION ---")

    # 4A. Custom C++ Text Box Detector
    final_edges = custom_edge_detector.get_canny_img(
        my_rotated_image, sigma=1.0, hight_threshold=0.20
    )
    img_to_segment = HoughTransform.conditional_rotation(final_edges)

    start = time.time()
    text_box_detector = TextBoxDetector.TextBoxDetector(img_to_segment)
    text_rows = text_box_detector.detect_symbol_boxes(
        density_threshold=6.2, pixel_threshold=1
    )
    custom_box_time = time.time() - start
    print(f"Custom C++ Boxes Time:     {custom_box_time:.4f}s")

    # 4B. Tesseract comparison
    img_for_tesseract = HoughTransform.conditional_rotation(my_rotated_image)

    # This function plots Custom vs Tesseract automatically and measures Tesseract's time internally
    text_analyzer.compare_symbol_boxes(
        img_for_tesseract, text_rows, custom_time=custom_box_time, show=False
    )

    print("\nCalculations complete! Displaying 4 benchmark windows...")
    print("Close the matplotlib windows to exit the script.")

    plt.show()


if __name__ == "__main__":
    run_benchmarks()
