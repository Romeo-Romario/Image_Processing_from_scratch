from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QScrollArea, QSlider, QGroupBox, QFormLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class HoughPage(QWidget):
    def __init__(self, logic_bridge):
        super().__init__()
        self.logic = logic_bridge
        self.layout = QHBoxLayout(self)

        # 1. Left Sidebar (Controls)
        self.controls = QGroupBox("Hough Parameters")
        self.controls.setFixedWidth(250)
        self.controls_layout = QFormLayout(self.controls)
        
        self.rho_slider = self._create_slider(1, 20, 9)
        self.controls_layout.addRow("Rho:", self.rho_slider)
        
        self.theta_slider = self._create_slider(0.001, 0.1, 0.0261)
        self.controls_layout.addRow("Theta:", self.theta_slider)

        self.btn_run = QPushButton("Run Deskew")
        self.btn_run.clicked.connect(self.run_hough)
        self.controls_layout.addRow(self.btn_run)
        
        self.layout.addWidget(self.controls)

        # 2. Right Area (Result Display)
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("Deskewed image will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def _create_slider(self, min_val, max_val, default):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 1000))
        slider.setMaximum(int(max_val * 1000))
        slider.setValue(int(default * 1000))
        return slider

    def run_hough(self):
        rho = self.rho_slider.value() / 1000.0
        theta = self.theta_slider.value() / 1000.0
        
        # We need edges first!
        edges = self.logic.processed_edges
        if edges is None:
            # If not run yet, run with defaults
            edges = self.logic.run_canny()
            
        if edges is not None:
            deskewed_np = self.logic.run_hough_deskew(edges, rho, theta)
            self.display_numpy_image(deskewed_np)

    def display_numpy_image(self, img_np):
        img_uint8 = (img_np * 255).astype(np.uint8)
        h, w = img_uint8.shape
        q_img = QImage(img_uint8.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.scroll_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setText("")
