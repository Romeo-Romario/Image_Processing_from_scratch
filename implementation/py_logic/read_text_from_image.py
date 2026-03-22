import torch
import torch.nn.functional as F
from autocorrect import Speller

# 1. Initialize the Ukrainian spell checker (fast=True optimizes it for speed)
spell_uk = Speller(lang="uk", fast=True)


def extract_and_read_row(
    deskewed_image, text_row_boxes, model, class_mapping, device, margin=5
):
    """
    Takes the deskewed image and the C++ bounding boxes, formats them for PyTorch,
    runs batch inference, and reconstructs the words.
    """
    if not text_row_boxes:
        return ""

    # 1. Sort boxes from left to right using the Point object's .x attribute!
    boxes = sorted(text_row_boxes, key=lambda b: b[0].x)

    char_tensors = []

    # --- PART A: CROP, PAD TO SQUARE, AND RESIZE ---
    for box in boxes:
        x1, y1 = int(box[0].x), int(box[0].y)
        x2, y2 = int(box[1].x), int(box[1].y)

        # Crop the raw character from the numpy array
        char_crop = deskewed_image[y1:y2, x1:x2]
        h, w = char_crop.shape

        if h == 0 or w == 0:
            continue

        char_tensor = torch.tensor(char_crop, dtype=torch.float32)

        # --- CRITICAL FIX 1: NORMALIZATION ---
        # Force the pixels into the 0.0 - 1.0 range that the model expects!
        if char_tensor.max() > 1.0:
            char_tensor = char_tensor / 255.0

        # Figure out how much padding is needed to make it a perfect square
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        # Add the squaring padding + your requested 5-pixel safe margin
        # Assuming your background is white, the pad_value must be 1.0 (white in normalized scale)
        char_tensor = F.pad(
            char_tensor,
            (
                pad_left + margin,
                pad_right + margin,
                pad_top + margin,
                pad_bottom + margin,
            ),
            value=1.0,
        )

        # Resize to exactly 64x64
        char_tensor = char_tensor.unsqueeze(0).unsqueeze(0)
        char_tensor = F.interpolate(
            char_tensor, size=(64, 64), mode="bilinear", align_corners=False
        )
        char_tensor = char_tensor.squeeze(0)  # Now it's [1, 64, 64]

        # --- CRITICAL FIX 2: COLOR INVERSION (CHECK THIS) ---
        # If your training dataset was WHITE text on a BLACK background, uncomment the line below!
        # char_tensor = 1.0 - char_tensor

        char_tensors.append(char_tensor)

    # --- PART B: BATCH INFERENCE (GPU) ---
    batch_tensor = torch.stack(char_tensors).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted_indices = torch.max(outputs, 1)

    predictions = predicted_indices.cpu().numpy()

    # --- PART C: CONNECT INTO WORDS (SPACE DETECTION) ---
    avg_char_width = sum([(b[1].x - b[0].x) for b in boxes]) / len(boxes)

    # 1. Lowered the space threshold to catch tighter word gaps
    space_threshold = avg_char_width * 0.40

    # 2. Dictionary to translate folder names back to real punctuation
    punctuation_fixes = {
        "dot": ".",
        "comma": ",",
        "hyphen": "-",
        "quote": "'",
        "semicolon": ";",
        "colon": ":",
        "question": "?",
        "exclamation": "!",
        "bracket_open": "(",
        "bracket_close": ")",
    }

    final_text = ""
    for i in range(len(predictions)):
        raw_char = class_mapping[predictions[i]]

        # Apply the punctuation fix (if it's not in the dictionary, keep the raw_char)
        predicted_char = punctuation_fixes.get(raw_char, raw_char)

        final_text += predicted_char

        # Space detection
        if i < len(predictions) - 1:
            current_box_right = boxes[i][1].x
            next_box_left = boxes[i + 1][0].x

            gap = next_box_left - current_box_right
            if gap > space_threshold:
                final_text += " "

    # --- PART D: THE SPELL CHECKER ---
    # The Speller automatically ignores punctuation and only fixes the words.
    # It will take your raw OCR string and return the polished Ukrainian text!
    corrected_text = spell_uk(final_text)

    return corrected_text
