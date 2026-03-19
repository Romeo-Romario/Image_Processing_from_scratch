import sys
import os
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from logic_bridge import LogicBridge
from pages.image_loader_page import ImageLoaderPage
from pages.canny_page import CannyPage
from pages.hough_page import HoughPage
from pages.segmentation_page import SegmentationPage

# Add pages for Hough and Segmentation later

def main():
    # Set high DPI scaling if needed (optional)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) if hasattr(Qt, 'AA_EnableHighDpiScaling') else None
    
    app = QApplication(sys.argv)
    
    # 1. Logic Bridge
    logic = LogicBridge()
    
    # 2. Main Window
    window = MainWindow(logic)
    
    # 3. Initialize Pages
    loader_page = ImageLoaderPage(logic)
    canny_page = CannyPage(logic)
    hough_page = HoughPage(logic)
    segmentation_page = SegmentationPage(logic)
    
    # 4. Add Pages to Content Stack
    window.add_page(loader_page)
    window.add_page(canny_page)
    window.add_page(hough_page)
    window.add_page(segmentation_page)
    
    # Start on loader page
    window.content_stack.setCurrentIndex(0)
    window.btn_load.setChecked(True)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    from PyQt5.QtCore import Qt # Import here to avoid early usage
    main()
