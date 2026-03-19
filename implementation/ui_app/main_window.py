from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QStackedWidget, QFrame, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self, logic_bridge):
        super().__init__()
        self.logic = logic_bridge
        self.setWindowTitle("OCR Pipeline Visualizer")
        self.resize(1200, 800)

        # Main horizontal layout: sidebar + content
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar (Navigation)
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("background-color: #2c3e50; color: white;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)

        title_label = QLabel("OCR Pipeline")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("padding: 20px 0;")
        sidebar_layout.addWidget(title_label)

        self.btn_load = self._create_nav_btn("1. Load Image")
        self.btn_canny = self._create_nav_btn("2. Canny Edges")
        self.btn_hough = self._create_nav_btn("3. Hough Deskew")
        self.btn_text = self._create_nav_btn("4. Text Detection")

        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addWidget(self.btn_canny)
        sidebar_layout.addWidget(self.btn_hough)
        sidebar_layout.addWidget(self.btn_text)

        # 2. Content Area (Stacked Widget)
        self.content_stack = QStackedWidget()
        
        # We will add pages later
        # self.content_stack.addWidget(Page1) ...

        # Combine
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content_stack)
        self.setCentralWidget(central_widget)

        # Signals
        self.btn_load.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.btn_canny.clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        self.btn_hough.clicked.connect(lambda: self.content_stack.setCurrentIndex(2))
        self.btn_text.clicked.connect(lambda: self.content_stack.setCurrentIndex(3))

    def _create_nav_btn(self, text):
        btn = QPushButton(text)
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 15px;
                text-align: left;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:checked {
                background-color: #3498db;
            }
        """)
        btn.setCheckable(True)
        # Ensure only one button is checked at a time (like a radio button group)
        btn.setAutoExclusive(True)
        return btn

    def add_page(self, widget):
        self.content_stack.addWidget(widget)
