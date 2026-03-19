import sys
import os
import numpy as np

# Adjust sys.path to find compiled logic modules
LOGIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logic"))

def add_to_path(path):
    if path not in sys.path:
        sys.path.append(path)

# Path to .pyd files (they are usually in the same dir as bindings.cpp or in build/)
add_to_path(os.path.join(LOGIC_ROOT, "edge_detection"))
add_to_path(os.path.join(LOGIC_ROOT, "hough_transform"))
add_to_path(os.path.join(LOGIC_ROOT, "text_box_detector"))

try:
    import EdgeDetector
    import HoughTransform
    import TextBoxDetector
except ImportError as e:
    print(f"Error importing C++ modules: {e}")
    # We will try to find them in build/lib* if not found in root
    # This is a fallback
    for module_dir in ["edge_detection", "hough_transform", "text_box_detector"]:
        build_dir = os.path.join(LOGIC_ROOT, module_dir, "build")
        if os.path.exists(build_dir):
            for root, dirs, files in os.walk(build_dir):
                if any(f.endswith(".pyd") for f in files):
                    add_to_path(root)
    
    import EdgeDetector
    import HoughTransform
    import TextBoxDetector

class LogicBridge:
    def __init__(self):
        self.canny = EdgeDetector.CannyEdgeDetector()
        self.hough = None
        self.text_detector = None
        self.original_image = None
        self.processed_edges = None
        self.deskewed_image = None
        self.deskewed_edges = None

    def set_original_image(self, image_np):
        """Expects normalized numpy array (float64, 0.0-1.0)"""
        self.original_image = image_np
        self.canny = EdgeDetector.CannyEdgeDetector(self.original_image)

    def run_canny(self, sigma=1.0, high=0.4, low=0.06):
        if self.original_image is None:
            return None
        self.processed_edges = self.canny.get_canny_img(self.original_image, sigma, high, low)
        return self.processed_edges

    def get_canny_intermediates(self):
        """Returns dict of intermediate results for visualization"""
        try:
            return {
                "grey": self.canny.get_grey_image(),
                "convolved": self.canny.get_convolved_image(),
                "gradients": self.canny.get_image_gradients(), # [dX, dY, Mag]
                "orientation": self.canny.get_image_gradient_orientation(), # [Ori, Rounded]
                "nms": self.canny.get_non_max_suppresion(),
                "thresholded": self.canny.get_thresholded_img(),
                "hysteresis": self.canny.get_hysteresis_img()
            }
        except Exception as e:
            print(f"Error getting Canny intermediates: {e}")
            return {}

    def run_hough_deskew(self, edges, rho=9, theta=0.0261):
        self.hough = HoughTransform.HoughTransform(edges, theta, rho)
        # We need the original image to rotate it
        # Note: deskew_image takes the uint8 or float image to rotate
        self.deskewed_image = self.hough.deskew_image(self.original_image, 2000.0, -np.pi, np.pi)
        return self.deskewed_image

    def run_text_detection(self, deskewed_image):
        # Usually we run Canny again on deskewed image before text detection
        self.deskewed_edges = self.canny.get_canny_img(deskewed_image, 1.0, 0.20, 0.06)
        # Apply conditional rotation if needed (logic from main.py)
        self.deskewed_edges = HoughTransform.conditional_rotation(self.deskewed_edges)
        
        self.text_detector = TextBoxDetector.TextBoxDetector(self.deskewed_edges)
        text_rows = self.text_detector.detect_symbol_boxes()
        return text_rows

    def get_text_detection_intermediates(self):
        if not self.text_detector:
            return {}
        return {
            "smoothed_f": self.text_detector.get_smoothed_img_f(),
            "extreme_points": self.text_detector.get_indexes_of_rows_extreame_points(),
            "clean_rows": self.text_detector.get_clean_text_rows()
        }
