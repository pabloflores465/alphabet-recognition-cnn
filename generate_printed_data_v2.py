#!/usr/bin/env python3
"""
Generate synthetic printed character images using the SAME preprocessing
pipeline that the webcam demo uses. This ensures consistency.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random

OUTPUT_DIR = Path(__file__).parent / "printed_dataset_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
FONT_DIRS = ["/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/"]

LETTERS = [chr(ord('A') + i) for i in range(26)]

# ─── PREPROCESSING (matches webcam_demo.py pipeline) ───────────────────
def preprocess_character(char_img_gray):
    """Preprocess a character image exactly like the webcam demo.
    char_img_gray: PIL Image or numpy array, grayscale, dark char on light bg
    Returns: numpy (28, 28) float32 [0-1] where 1 = character
    """
    if isinstance(char_img_gray, Image.Image):
        gray = np.array(char_img_gray)
    else:
        gray = char_img_gray.copy()

    # Invert: char should be bright on dark
    gray = 255 - gray

    # Threshold (OTSU)
    if gray.std() > 20:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contour to center
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(max(w, h) * 0.15)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2 * pad)
        h = min(binary.shape[0] - y, h + 2 * pad)
        char = binary[y:y+h, x:x+w]
    else:
        char = binary

    if char.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Resize maintaining aspect ratio
    h, w = char.shape
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    char_resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28 canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_resized

    return canvas.astype(np.float32) / 255.0


# ─── FIND FONTS ────────────────────────────────────────────────────────
def find_fonts():
    found = []
    for font_dir in FONT_DIRS:
        if not os.path.exists(font_dir):
            continue
        for fname in os.listdir(font_dir):
            fpath = os.path.join(font_dir, fname)
            if fname.endswith('.ttf') or fname.endswith('.otf') or fname.endswith('.ttc'):
                # Skip CJK
                if any(cjk in fname for cjk in ['ヒラキ', 'Gothic', 'Myungjo', 'SD', 'PingFang', 'Heiti']):
                    continue
                # Quick test
                try:
                    font = ImageFont.truetype(fpath, 20)
                    found.append(fpath)
                except:
                    pass
    print(f"Found {len(found)} fonts")
    return found


# ─── GENERATE ───────────────────────────────────────────────────────────
import cv2

def generate_printed(fonts, samples_per_font=30):
    """Generate printed character images using webcam-compatible pipeline."""
    all_images = []
    all_labels = []

    for letter_idx, letter in enumerate(LETTERS):
        letter_imgs = []
        for font_path in fonts:
            for _ in range(samples_per_font):
                # Random size
                size = random.choice([18, 20, 22, 24, 26, 28, 30, 34])
                try:
                    font = ImageFont.truetype(font_path, size)
                except:
                    continue

                # Create large canvas
                canvas_size = random.choice([56, 64, 72, 80])
                img = Image.new("L", (canvas_size, canvas_size), color=255)
                draw = ImageDraw.Draw(img)

                # Get text bounding box
                bbox = draw.textbbox((0, 0), letter, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Center with jitter
                jx = random.randint(-3, 3)
                jy = random.randint(-3, 3)
                x = (canvas_size - tw) // 2 + jx - bbox[0]
                y = (canvas_size - th) // 2 + jy - bbox[1]

                # Random tilt
                if random.random() < 0.3:
                    angle = random.uniform(-5, 5)
                    tmp = Image.new("L", (canvas_size+10, canvas_size+10), color=255)
                    tmp_draw = ImageDraw.Draw(tmp)
                    tmp_draw.text((x+5, y+5), letter, font=font, fill=0)
                    tmp = tmp.rotate(angle, fillcolor=255)
                    img = tmp.crop((5, 5, canvas_size+5, canvas_size+5))
                else:
                    draw.text((x, y), letter, font=font, fill=0)

                # Apply variations
                # Blur
                if random.random() < 0.15:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

                # Add faint background noise
                if random.random() < 0.2:
                    arr = np.array(img, dtype=np.float32)
                    noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
                    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)

                # Contrast variation
                if random.random() < 0.3:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.4))

                # Preprocess using same pipeline as webcam
                processed = preprocess_character(img)

                letter_imgs.append(processed)

        if letter_imgs:
            # Keep all for this letter
            all_images.extend(letter_imgs)
            all_labels.extend([letter_idx] * len(letter_imgs))
            print(f"  {letter}: {len(letter_imgs)} images")

    return all_images, all_labels


# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PRINTED CHARACTER GENERATOR v2")
    print("  (Webcam-compatible preprocessing)")
    print("=" * 60)

    fonts = find_fonts()
    random.seed(42)
    selected = random.sample(fonts, min(30, len(fonts)))
    print(f"Using {len(selected)} fonts")

    images, labels = generate_printed(selected, samples_per_font=25)

    X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    output_path = OUTPUT_DIR / "printed_letters_v2.npz"
    np.savez_compressed(output_path, X=X, y=y)
    print(f"\nSaved {len(X)} images to {output_path}")


if __name__ == "__main__":
    main()
