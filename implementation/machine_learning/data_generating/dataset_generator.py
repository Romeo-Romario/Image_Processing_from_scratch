import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import uuid
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGES_DIR = r"D:\Source\Diplom\tryouts\book_images"
DATASET_DIR = r"D:\Source\Diplom\tryouts\tryout2_image_deskweing\implementation\machine_learning\dataset"
pytesseract.pytesseract.tesseract_cmd = r"D:\Libaries\Tesseract\tesseract.exe"

# Lowered confidence. 75 is still high enough to avoid garbage, but low enough to actually get data!
MIN_CONFIDENCE = 75

SAFE_DIR_NAMES = {
    ".": "dot",
    ",": "comma",
    ":": "colon",
    ";": "semicolon",
    "!": "exclamation",
    "?": "question_mark",
    "-": "hyphen",
    "'": "apostrophe",
    '"': "quote",
    "(": "bracket_open",
    ")": "bracket_close",
}


def get_safe_label(text):
    text = text.strip().lower()
    if not text:
        return None
    if text in SAFE_DIR_NAMES:
        return SAFE_DIR_NAMES[text]
    if not text.isalnum():
        return None
    return text


def ensure_portrait(image):
    h, w = image.shape[:2]
    if w > h:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def process_book():
    os.makedirs(DATASET_DIR, exist_ok=True)
    image_files = [
        f
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Found {len(image_files)} images. Starting dataset generation...")

    total_extracted = 0

    for filename in tqdm(image_files, desc="Processing Pages"):
        img_path = os.path.join(IMAGES_DIR, filename)

        # FIX 1: Safe read for Windows (in case your image names ever contain Cyrillic)
        with open(img_path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        gray = cv2.imdecode(chunk_arr, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        gray = ensure_portrait(gray)
        h, w = gray.shape

        custom_config = r"-l ukr --psm 3"

        # --- PASS 1: Find Confident Words ---
        data = pytesseract.image_to_data(
            gray, config=custom_config, output_type=Output.DICT
        )

        confident_zones = []
        for i in range(len(data["text"])):
            conf = int(data["conf"][i])
            text = data["text"][i].strip()

            if conf >= MIN_CONFIDENCE and len(text) > 0:
                bx = data["left"][i]
                by = data["top"][i]
                bw = data["width"][i]
                bh = data["height"][i]
                confident_zones.append((bx, by, bx + bw, by + bh))

        # --- PASS 2: Extract Characters ---
        boxes = pytesseract.image_to_boxes(gray, config=custom_config)

        chars_this_page = 0

        for b in boxes.splitlines():
            b = b.split(" ")
            char = b[0]
            safe_label = get_safe_label(char)

            if not safe_label:
                continue

            left, bottom, right, top = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            x1 = left
            x2 = right
            y1 = h - top
            y2 = h - bottom

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                continue

            cx = x1 + (x2 - x1) / 2
            cy = y1 + (y2 - y1) / 2

            is_confident = False
            for bx1, by1, bx2, by2 in confident_zones:
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    is_confident = True
                    break

            if is_confident:
                char_img = gray[y1:y2, x1:x2]
                char_img_padded = cv2.copyMakeBorder(
                    char_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255
                )

                label_dir = os.path.join(DATASET_DIR, safe_label)
                os.makedirs(label_dir, exist_ok=True)
                save_path = os.path.join(label_dir, f"{uuid.uuid4().hex}.png")

                # FIX 2: Safe write for Windows Cyrillic paths
                # Instead of cv2.imwrite, we encode it to a memory buffer and write it using standard Python
                is_success, im_buf_arr = cv2.imencode(".png", char_img_padded)
                if is_success:
                    im_buf_arr.tofile(save_path)
                    chars_this_page += 1
                    total_extracted += 1

        tqdm.write(
            f"Page {filename}: Extracted {chars_this_page} confident characters."
        )

    print(
        f"\nDONE! Successfully extracted a total of {total_extracted} characters across all pages."
    )


if __name__ == "__main__":
    process_book()
