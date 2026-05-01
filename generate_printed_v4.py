#!/usr/bin/env python3
"""
Generate HIGH-QUALITY printed character dataset v4.
- Always centered via weighted centroid
- Quality check: rejects empty/misaligned images
- Same preprocessing pipeline as webcam
"""

import os, sys, random, cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUTPUT_DIR = Path(__file__).parent / "printed_dataset_v4"
OUTPUT_DIR.mkdir(exist_ok=True)
IMG_SIZE = 28
FONT_DIRS = ["/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/"]
LETTERS = [chr(ord('A')+i) for i in range(26)]


def find_fonts():
    found = []
    for d in FONT_DIRS:
        if not os.path.exists(d): continue
        for fname in os.listdir(d):
            fp = os.path.join(d, fname)
            if not (fname.endswith('.ttf') or fname.endswith('.otf') or fname.endswith('.ttc')): continue
            if any(cjk in fname for cjk in ['ヒラキ','Gothic','Myungjo','SD','PingFang','Heiti','AppleGothic']): continue
            try:
                ImageFont.truetype(fp, 20); found.append(fp)
            except: pass
    print(f"Found {len(found)} fonts")
    return found


def preprocess_robust(char_img_gray):
    """
    Robust preprocessing: center character via center of mass, resize to 28x28.
    char_img_gray: PIL Image, dark char on light bg.
    Returns: (28,28) float32 [0-1] or None if quality check fails.
    """
    gray = np.array(char_img_gray).astype(np.float32)
    gray = 255.0 - gray        # invert: char = bright
    gray = gray / 255.0

    # Center of mass
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    mass = gray.sum()
    if mass < 0.5:
        return None  # too faint
    cy = (ys * gray).sum() / mass
    cx = (xs * gray).sum() / mass

    # Bounding box around center of mass
    # Find extent by thresholding > mean
    thresh = gray.mean() * 0.8
    mask = gray > thresh
    if mask.sum() < 4:
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin = np.argmax(rows)
    ymax = h - 1 - np.argmax(rows[::-1])
    xmin = np.argmax(cols)
    xmax = w - 1 - np.argmax(cols[::-1])

    pad = max(2, int(max(ymax-ymin, xmax-xmin) * 0.25))
    ymin = max(0, ymin - pad)
    ymax = min(h, ymax + pad + 1)
    xmin = max(0, xmin - pad)
    xmax = min(w, xmax + pad + 1)

    char = gray[ymin:ymax, xmin:xmax]
    if char.size < 4:
        return None

    # Resize maintaining aspect ratio
    ch, cw = char.shape
    scale = IMG_SIZE / max(ch, cw)
    new_h = max(1, int(ch * scale))
    new_w = max(1, int(cw * scale))
    char_rs = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_rs

    # Quality check: character must be near center
    mass2 = canvas.sum()
    if mass2 < 0.3:
        return None

    ys2, xs2 = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy2 = (ys2 * canvas).sum() / mass2
    cx2 = (xs2 * canvas).sum() / mass2
    # Center must be within central 60% of the canvas
    if abs(cy2 - IMG_SIZE/2) > IMG_SIZE*0.30 or abs(cx2 - IMG_SIZE/2) > IMG_SIZE*0.30:
        return None

    return canvas


def generate(fonts, per_font=25):
    all_imgs, all_lbls = [], []
    for li, letter in enumerate(LETTERS):
        imgs = []
        for fp in fonts:
            for _ in range(per_font):
                size = random.choice([20, 24, 28, 32, 36, 42])
                try:
                    font = ImageFont.truetype(fp, size)
                except: continue

                cs = random.choice([56, 64, 72, 80])
                img = Image.new("L", (cs, cs), 255)
                draw = ImageDraw.Draw(img)

                bbox = draw.textbbox((0,0), letter, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                jx, jy = random.randint(-2,2), random.randint(-2,2)
                x = (cs-tw)//2 + jx - bbox[0]
                y = (cs-th)//2 + jy - bbox[1]

                if random.random() < 0.25:
                    ang = random.uniform(-5,5)
                    tmp = Image.new("L", (cs+10,cs+10), 255)
                    tmpd = ImageDraw.Draw(tmp)
                    tmpd.text((x+5,y+5), letter, font=font, fill=0)
                    tmp = tmp.rotate(ang, fillcolor=255)
                    img = tmp.crop((5,5,cs+5,cs+5))
                else:
                    draw.text((x,y), letter, font=font, fill=0)

                if random.random() < 0.15:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2,0.6)))
                if random.random() < 0.2:
                    arr = np.array(img,dtype=np.float32)
                    arr = np.clip(arr + np.random.normal(0,random.uniform(2,6),arr.shape),0,255).astype(np.uint8)
                    img = Image.fromarray(arr)
                if random.random() < 0.3:
                    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8,1.3))

                processed = preprocess_robust(img)
                if processed is not None:
                    imgs.append(processed)

        all_imgs.extend(imgs)
        all_lbls.extend([li]*len(imgs))
        print(f"  {letter}: {len(imgs)} images (rejected {per_font*len(fonts)-len(imgs)})")

    return all_imgs, all_lbls


def main():
    print("="*60)
    print("  PRINTED DATASET v4 — Quality-controlled")
    print("="*60)
    fonts = find_fonts()
    random.seed(42)
    selected = random.sample(fonts, min(40, len(fonts)))
    print(f"Using {len(selected)} fonts")

    images, labels = generate(selected, per_font=30)
    X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype=np.int32)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    out = OUTPUT_DIR / "printed_v4.npz"
    np.savez_compressed(out, X=X, y=y)
    print(f"\nSaved {len(X):,} quality images → {out}")

    # Stats
    for i in range(26):
        cnt = (y == i).sum()
        print(f"  {LETTERS[i]}: {cnt}")


if __name__ == "__main__":
    main()
