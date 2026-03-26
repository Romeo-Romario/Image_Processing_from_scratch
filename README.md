# Custom OCR and Document Analysis Pipeline

This repository contains a high-performance computer vision pipeline built from scratch to extract, deskew, and segment text from document images. It is developed as part of a diploma project focusing on algorithmic image processing and text extraction.

The core algorithms are written in **C++** for maximum execution speed and multithreaded performance, while the orchestration, data handling, and visualization are managed in **Python** using **pybind11** to bridge the two languages.

## Current Features

The project currently implements a full pre-processing and segmentation pipeline without relying on external deep learning models for the spatial extraction phase:

* **Edge Detection (`EdgeDetector`)**: A custom Canny Edge Detection implementation to identify document boundaries and text shapes.
  <img width="1769" height="1064" alt="image" src="https://github.com/user-attachments/assets/891f6345-53ed-4683-92a0-681c39cda3a8" />

* **Image Deskewing (`HoughTransform`)**: Algorithmic rotation correction using the Hough Transform to automatically straighten tilted documents.
  
  <img width="160" height="435" alt="image" src="https://github.com/user-attachments/assets/0297db8b-020f-4987-a401-9e0fdec91cd8" />
  <img width="794" height="623" alt="image" src="https://github.com/user-attachments/assets/424a530c-39ca-4522-ba1c-c8b435696627" />

* **Text Row Segmentation (`TextBoxDetector`)**: 
    * Calculates horizontal projection profiles.
    * Applies 1D Gaussian smoothing to filter noise.
    * Identifies peaks and valleys to accurately slice the document into individual lines of text.
    <img width="2526" height="1402" alt="image" src="https://github.com/user-attachments/assets/9378f5b2-6533-4d43-8133-e4f028c099da" />

* **Character / Symbol Segmentation**:
    * Uses vertical projection profiles (zero-crossing) for rapid baseline character separation.
    * Calculates average symbol widths to identify overlapping or merged characters.
    * Employs an iterative **Depth-First Search (DFS)** algorithm as a fallback to trace connected pixel components for complex or fragmented symbols.
    * Normalizes (tight-crops) bounding boxes to perfectly wrap the individual characters.
    <img width="2550" height="304" alt="image" src="https://github.com/user-attachments/assets/aad22640-588a-4b56-8219-9abb128b4d18" />

* **Tesseract Benchmarking**: Includes built-in visualizers to run the custom segmentation alongside Tesseract OCR side-by-side for direct accuracy and performance comparison.
<img width="2345" height="1429" alt="image" src="https://github.com/user-attachments/assets/7d6f4542-e49d-4098-8c3f-8e31305e77f1" />


* **Final result**
<img width="1223" height="856" alt="image" src="https://github.com/user-attachments/assets/9e01aa24-af5e-4e6c-ac67-c61584c0ad75" />


## Tech Stack

* **Core Logic**: C++17 (Standard Library, custom multithreading)
* **Bindings**: `pybind11` (for seamless C++ to Python interoperability)
* **Scripting**: Python 3
* **Visualization**: NumPy, Matplotlib

## Project Structure

```text
├── implementation/
│   ├── main.py                     # Main execution script and visualization
│   ├── images_of_book/             # Test datasets
│   ├── logic/
│   │   ├── edge_detection/         # C++ Canny Edge Detector module
│   │   ├── hough_transform/        # C++ Deskewing module
│   │   ├── text_box_detector/      # C++ Row and Symbol segmentation module
│   │   └── additional_modules/     # C++ Utilities (Matrix <-> NumPy converters, Threading)
│   └── py_logic/                   # Python visualization and analysis scripts
```

Build and Installation
Because the heavy lifting is done in C++, the modules must be compiled into Python-readable binaries (.pyd files on Windows) before running the main script.

Install Python Dependencies:
Bash
```text
pip install numpy matplotlib opencv-python Pillow pytesseract
pip install pybind11
```
(Note: To use the Tesseract comparison feature, the Tesseract OCR engine must be installed on your system).

Compile the C++ Modules:
Navigate to each module directory (edge_detection, hough_transform, text_box_detector) and build the pybind11 extensions:

Bash
```text
python setup.py build_ext --inplace
pybind11-stubgen EdgeDetector -o . 
```

Usage
Run the main orchestrator script to process an image through the pipeline and visualize the segmented bounding boxes:

Bash
```
cd implementation
python main.py
```
The script will output the timing for the C++ multithreaded operations and display an interactive Matplotlib window comparing the custom bounding boxes (Red) against Tesseract's bounding boxes (Blue).


**Future Work: Symbol Recognition**
The pipeline successfully isolates clean, normalized bounding boxes for individual characters. The next phase of this project is to implement Model Training for Symbol Detection.
Future updates will include:

 * Extracting the pixel matrices from the normalized SymbolBox structs.
 * Building a dataset of these segmented characters.
 * Training a Machine Learning model (e.g., a Convolutional Neural Network) to classify these matrices into actual textual string data, completing the full OCR pipeline.
