from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QScrollArea, QSlider, QGroupBox, QFormLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2

class CannyPage(QWidget):
    def __init__(self, logic_bridge):
        super().__init__()
        self.logic = logic_bridge
        self.layout = QHBoxLayout(self)

        # 1. Left Sidebar (Controls)
        self.controls = QGroupBox("Canny Parameters")
        self.controls.setFixedWidth(250)
        self.controls_layout = QFormLayout(self.controls)
        
        # Sigma Slider
        self.sigma_slider = self._create_slider(0.1, 5.0, 1.0)
        self.controls_layout.addRow("Sigma:", self.sigma_slider)
        
        # High Threshold Slider
        self.high_slider = self._create_slider(0.01, 1.0, 0.4)
        self.controls_layout.addRow("High T:", self.high_slider)
        
        # Low Threshold Slider
        self.low_slider = self._create_slider(0.01, 1.0, 0.06)
        self.controls_layout.addRow("Low T:", self.low_slider)

        self.btn_run = QPushButton("Run Canny")
        self.btn_run.clicked.connect(self.run_canny)
        self.controls_layout.addRow(self.btn_run)
        
        self.layout.addWidget(self.controls)

        # 2. Right Area (Result Display)
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("Result will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def _create_slider(self, min_val, max_val, default):
        # QSlider works with integers, so we scale by 100
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(int(default * 100))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        return slider

    def run_canny(self):
        sigma = self.sigma_slider.value() / 100.0
        high = self.high_slider.value() / 100.0
        low = self.low_slider.value() / 100.0
        
        result_np = self.logic.run_canny(sigma, high, low)
        if result_np is not None:
            self.display_numpy_image(result_np)

    def display_numpy_image(self, img_np):
        # Convert float64 (0-1) to uint8 (0-255)
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        h, w = img_uint8.shape
        q_img = QImage(img_uint8.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.scroll_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setText("")
