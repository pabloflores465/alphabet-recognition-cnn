#!/usr/bin/env python3
"""
Generate printed character images v3 — with grayscale preservation.
No aggressive thresholding. Uses same centering pipeline but keeps gray levels.
This matches what the webcam actually captures better.
"""

import os, sys, random, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUTPUT_DIR = Path(__file__).parent / "printed_dataset_v3"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
FONT_DIRS = ["/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/"]
LETTERS = [chr(ord('A') + i) for i in range(26)]


def find_fonts():
    found = []
    for font_dir in FONT_DIRS:
        if not os.path.exists(font_dir): continue
        for fname in os.listdir(font_dir):
            if fname.endswith('.ttf') or fname.endswith('.otf') or fname.endswith('.ttc'):
                if any(cjk in fname for cjk in ['ヒラキ', 'Gothic', 'Myungjo', 'SD', 'PingFang', 'Heiti']):
                    continue
                try:
                    ImageFont.truetype(os.path.join(font_dir, fname), 20)
                    found.append(os.path.join(font_dir, fname))
                except: pass
    print(f"Found {len(found)} fonts")
    return found


def preprocess_grayscale(char_img_gray):
    """Preprocess keeping grayscale info. Robust centering via center of mass.
    char_img_gray: PIL Image, dark char on light bg
    Returns: numpy (28, 28) float32 [0-1] where 1 = character pixel"""
    gray = np.array(char_img_gray).astype(np.float32)

    # Invert: dark char (0) → bright char (1), light bg (255) → dark bg (0)
    gray = 255.0 - gray
    gray = gray / 255.0

    # Find center of mass for centering
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    mass = gray.sum()
    if mass > 0:
        cy = (ys * gray).sum() / mass
        cx = (xs * gray).sum() / mass
    else:
        cy, cx = h/2, w/2

    # Find bounding box around center of mass
    # Use pixels above mean as character mask
    threshold = max(gray.mean() * 1.1, 0.02)
    mask = gray > threshold
    
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.argmax(rows), h - 1 - np.argmax(rows[::-1])
        xmin, xmax = np.argmax(cols), w - 1 - np.argmax(cols[::-1])
        
        # Add padding
        pad = max(1, int(max(ymax-ymin, xmax-xmin) * 0.2))
        ymin = max(0, ymin - pad)
        ymax = min(h, ymax + pad)
        xmin = max(0, xmin - pad)
        xmax = min(w, xmax + pad)
        
        char = gray[ymin:ymax, xmin:xmax]
    else:
        char = gray

    if char.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Resize maintaining aspect ratio to fit in 28x28
    ch, cw = char.shape
    scale = IMG_SIZE / max(ch, cw)
    new_h = max(1, int(ch * scale))
    new_w = max(1, int(cw * scale))
    
    # Use INTER_AREA for downscaling, INTER_CUBIC for upscaling
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    char_resized = cv2.resize(char, (new_w, new_h), interpolation=interp)

    # Center in 28x28 canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_resized

    return canvas


def generate(fonts, samples_per_font=30):
    all_imgs, all_lbls = [], []
    for letter_idx, letter in enumerate(LETTERS):
        imgs = []
        for font_path in fonts:
            for _ in range(samples_per_font):
                size = random.choice([18, 22, 26, 30, 34, 40])
                try:
                    font = ImageFont.truetype(font_path, size)
                except: continue

                cs = random.choice([56, 64, 72, 80])
                img = Image.new("L", (cs, cs), color=255)
                draw = ImageDraw.Draw(img)

                bbox = draw.textbbox((0, 0), letter, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                jx, jy = random.randint(-3, 3), random.randint(-3, 3)
                x = (cs-tw)//2 + jx - bbox[0]
                y = (cs-th)//2 + jy - bbox[1]

                if random.random() < 0.3:
                    angle = random.uniform(-6, 6)
                    tmp = Image.new("L", (cs+10, cs+10), 255)
                    tmp_draw = ImageDraw.Draw(tmp)
                    tmp_draw.text((x+5, y+5), letter, font=font, fill=0)
                    tmp = tmp.rotate(angle, fillcolor=255)
                    img = tmp.crop((5, 5, cs+5, cs+5))
                else:
                    draw.text((x, y), letter, font=font, fill=0)

                # Light blur
                if random.random() < 0.15:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

                # Light noise
                if random.random() < 0.2:
                    arr = np.array(img, dtype=np.float32)
                    noise = np.random.normal(0, random.uniform(2, 6), arr.shape)
                    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)

                # Contrast
                if random.random() < 0.3:
                    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

                processed = preprocess_grayscale(img)
                imgs.append(processed)

        all_imgs.extend(imgs)
        all_lbls.extend([letter_idx]*len(imgs))
        print(f"  {letter}: {len(imgs)} images")

    return all_imgs, all_lbls


def main():
    print("="*60)
    print("  PRINTED DATA v3 — Grayscale preserved")
    print("="*60)

    fonts = find_fonts()
    random.seed(42)
    selected = random.sample(fonts, min(35, len(fonts)))
    print(f"Using {len(selected)} fonts")

    images, labels = generate(selected, samples_per_font=30)

    X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    output_path = OUTPUT_DIR / "printed_letters_v3.npz"
    np.savez_compressed(output_path, X=X, y=y)
    print(f"\nSaved {len(X)} images to {output_path}")


if __name__ == "__main__":
    main()
