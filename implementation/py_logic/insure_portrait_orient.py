import cv2
import numpy as np


def conditional_rotate(img: np.ndarray) -> np.ndarray:
    """
    Rotates the image 90 degrees clockwise if rows < cols (Landscape).
    Returns the rotated image or the original image.
    """
    # img.shape is (rows/height, cols/width, channels)
    rows, cols = img.shape[:2]

    # Condition: If Height < Width
    if rows < cols:
        print(f"Rotating image: Rows({rows}) < Cols({cols}) -> Rotating 90° Clockwise")
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    print(f"Skipping rotation: Rows({rows}) >= Cols({cols})")
    return img
