import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2 as cv
import typing as tp


# Funtion to add lines to an image
def draw_lines(
    img: np.ndarray,
    lines: np.ndarray,
    color: tp.List[int] = [0, 0, 255],
    thickness: int = 1,
) -> tp.Tuple[np.ndarray]:

    if len(img.shape) == 2:
        new_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        new_image = np.copy(img)

    empty_image = np.zeros(img.shape[:2])

    if len(lines.shape) == 1:
        lines = lines[None, ...]

    print("LOGIC OF LINE TRANSFORM")
    for rho, theta in lines:
        if rho:
            print(f"rho: {rho} theta: {theta}")
            x0 = polar2cartesian(rho, theta)
            print(f"x0: {x0}")
            direction = np.array([x0[1], -x0[0]])
            print(f"direction: {direction}")
            pt1 = np.round(x0 + 1000 * direction).astype(int)
            print(f"pt1: {pt1}")
            pt2 = np.round(x0 - 1000 * direction).astype(int)
            print(f"pt2: {pt2}")
            empty_image = cv.line(
                img=empty_image, pt1=pt1, pt2=pt2, color=255, thickness=thickness
            )
        else:
            x0 = polar2cartesian(rho, theta)
            direction = np.array([x0[1], -x0[0]])
            pt1 = np.round(x0 + 1000 * direction).astype(int)
            pt2 = np.round(x0 - 1000 * direction).astype(int)
            empty_image = cv.line(
                img=empty_image, pt1=pt1, pt2=pt2, color=255, thickness=thickness
            )

    # Keep lower part of each line until intersection
    # mask_lines = empty_image != 0
    # min_diff = np.inf
    # max_line = 0
    # for i in range(mask_lines.shape[0]):
    #     line = mask_lines[i]
    #     indices = np.argwhere(line)
    #     if indices[-1] - indices[0] < min_diff:
    #         min_diff = indices[-1] - indices[0]
    #         max_line = i

    # mask_boundaries = np.zeros_like(empty_image)
    # mask_boundaries[max_line:] = 1
    # mask = (mask_lines * mask_boundaries).astype(bool)

    new_image[empty_image > 0] = np.array(color)

    return new_image, empty_image


# Function to perform the conversion between polar and cartesian coordinates
def polar2cartesian(
    radius: np.ndarray, angle: np.ndarray, cv2_setup: bool = True
) -> np.ndarray:
    return radius * np.array([np.sin(angle), np.cos(angle)])
