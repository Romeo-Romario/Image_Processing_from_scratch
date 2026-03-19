from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QScrollArea, QGroupBox, QFormLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect
import numpy as np

class SegmentationPage(QWidget):
    def __init__(self, logic_bridge):
        super().__init__()
        self.logic = logic_bridge
        self.layout = QHBoxLayout(self)

        # 1. Left Sidebar (Controls)
        self.controls = QGroupBox("Segmentation Controls")
        self.controls.setFixedWidth(250)
        self.controls_layout = QFormLayout(self.controls)
        
        self.btn_run = QPushButton("Run Text Detection")
        self.btn_run.clicked.connect(self.run_segmentation)
        self.controls_layout.addRow(self.btn_run)
        
        self.layout.addWidget(self.controls)

        # 2. Right Area (Result Display)
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("Segmented text will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)
        
        self.text_rows = []

    def run_segmentation(self):
        deskewed_img = self.logic.deskewed_image
        if deskewed_img is None:
            # Fallback to original if deskew not run
            deskewed_img = self.logic.original_image
            
        if deskewed_img is not None:
            self.text_rows = self.logic.run_text_detection(deskewed_img)
            self.display_numpy_with_boxes(deskewed_img, self.text_rows)

    def display_numpy_with_boxes(self, img_np, text_rows):
        img_uint8 = (img_np * 255).astype(np.uint8)
        h, w = img_uint8.shape
        
        # Convert to color QImage for red boxes
        q_img = QImage(img_uint8.data, w, h, w, QImage.Format_Grayscale8).convert(QImage.Format_RGB32)
        
        painter = QPainter(q_img)
        pen = QPen(QColor(255, 0, 0)) # Red
        pen.setWidth(2)
        painter.setPen(pen)
        
        for row in text_rows:
            # Draw row boundaries or symbol boxes
            for top_left, bottom_right in row.symbols_limits:
                rect = QRect(top_left.x, top_left.y, 
                             bottom_right.x - top_left.x, 
                             bottom_right.y - top_left.y)
                painter.drawRect(rect)
        
        painter.end()
        
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.scroll_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setText("")
