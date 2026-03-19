from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image

class ImageLoaderPage(QWidget):
    def __init__(self, logic_bridge):
        super().__init__()
        self.logic = logic_bridge
        self.layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Image...")
        self.btn_browse.clicked.connect(self.browse_image)
        toolbar.addWidget(self.btn_browse)
        toolbar.addStretch()
        self.layout.addLayout(toolbar)

        # Image Display Area
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("No image loaded.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        # 1. Load with PIL for numpy conversion
        pil_img = Image.open(file_path).convert("L")
        img_np = np.array(pil_img, dtype=np.float64) / 255.0
        
        # 2. Update logic bridge
        self.logic.set_original_image(img_np)
        
        # 3. Update UI
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap.scaled(
            self.scroll_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_label.setText("") # Clear "No image loaded" text
