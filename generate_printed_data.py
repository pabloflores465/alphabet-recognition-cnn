#!/usr/bin/env python3
"""
Generate synthetic printed alphabet character images using macOS fonts.
Creates a diverse dataset for training a robust A-Z classifier
that works for both printed and handwritten characters.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random

# ─── CONFIG ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "printed_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28
FONT_DIRS = [
    "/System/Library/Fonts/",
    "/System/Library/Fonts/Supplemental/",
    "/Library/Fonts/",
]

# English-friendly fonts (no CJK, no symbols-only)
FONT_WHITELIST = [
    "Arial.ttf", "Arial Bold.ttf", "Arial Bold Italic.ttf", "Arial Italic.ttf",
    "Arial Narrow.ttf", "Arial Narrow Bold.ttf",
    "Arial Rounded Bold.ttf",
    "Helvetica.ttc", "HelveticaNeue.ttc",
    "Times New Roman.ttf", "Times New Roman Bold.ttf",
    "Times New Roman Bold Italic.ttf", "Times New Roman Italic.ttf",
    "Georgia.ttf", "Georgia Bold.ttf", "Georgia Bold Italic.ttf", "Georgia Italic.ttf",
    "Verdana.ttf", "Verdana Bold.ttf", "Verdana Bold Italic.ttf", "Verdana Italic.ttf",
    "Courier New.ttf", "Courier New Bold.ttf",
    "AmericanTypewriter.ttc",
    "Andale Mono.ttf",
    "Apple Chancery.ttf",
    "Baskerville.ttc",
    "BigCaslon.ttf",
    "Brush Script.ttf",
    "Chalkboard.ttc",
    "ChalkboardSE.ttc",
    "Chalkduster.ttf",
    "Cochin.ttc",
    "Comic Sans MS.ttf",
    "Copperplate.ttc",
    "Didot.ttc",
    "Futura.ttc",
    "GillSans.ttc",
    "Herculanum.ttf",
    "Hoefler Text.ttc",
    "Impact.ttf",
    "MarkerFelt.ttc",
    "Noteworthy.ttc",
    "Optima.ttc",
    "Palatino.ttc",
    "Papyrus.ttc",
    "Phosphate.ttc",
    "Rockwell.ttc",
    "SignPainter.ttc",
    "Skia.ttf",
    "SnellRoundhand.ttc",
    "SuperClarendon.ttc",
    "Trebuchet MS.ttf",
    "Trattatello.ttf",
    "Zapfino.ttf",
]

LETTERS = [chr(ord('A') + i) for i in range(26)]

# ─── FIND FONTS ────────────────────────────────────────────────────────
def find_fonts():
    """Find available fonts on the system."""
    found = []
    for font_dir in FONT_DIRS:
        if not os.path.exists(font_dir):
            continue
        for fname in FONT_WHITELIST:
            fpath = os.path.join(font_dir, fname)
            if os.path.exists(fpath):
                found.append(fpath)
    # Also check .ttc files (font collections)
    for font_dir in FONT_DIRS:
        if not os.path.exists(font_dir):
            continue
        for fname in os.listdir(font_dir):
            if fname.endswith('.ttf') or fname.endswith('.otf'):
                fpath = os.path.join(font_dir, fname)
                # Skip CJK fonts by name
                if any(cjk in fname for cjk in ['ヒラキ', 'AppleGothic', 'AppleMyungjo', 'AppleSD']):
                    continue
                if fpath not in found:
                    # Quick test if it can render Latin chars
                    try:
                        test_font = ImageFont.truetype(fpath, 20)
                        found.append(fpath)
                    except:
                        pass

    print(f"Found {len(found)} fonts")
    return found


# ─── GENERATE IMAGES ───────────────────────────────────────────────────
def generate_for_letter(letter, letter_idx, fonts, samples_per_font=100):
    """Generate synthetic printed images for one letter."""
    images = []
    labels = []

    for font_path in fonts:
        try:
            # Try multiple sizes per font for variety
            sizes = random.choices([14, 16, 18, 20, 22], k=samples_per_font)
        except:
            continue

        for size in sizes:
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                continue

            # Create larger image then downscale for better quality
            temp_size = 56
            img = Image.new("L", (temp_size, temp_size), color=255)
            draw = ImageDraw.Draw(img)

            # Get character bounding box
            bbox = draw.textbbox((0, 0), letter, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Random position jitter
            jitter_x = random.randint(-3, 3)
            jitter_y = random.randint(-3, 3)
            x = (temp_size - tw) // 2 + jitter_x
            y = (temp_size - th) // 2 + jitter_y - bbox[1]

            # Random rotation
            if random.random() < 0.3:
                angle = random.uniform(-8, 8)
                img_rot = Image.new("L", (temp_size + 10, temp_size + 10), color=255)
                draw_rot = ImageDraw.Draw(img_rot)
                draw_rot.text((5 + x, 5 + y), letter, font=font, fill=0)
                img = img_rot.rotate(angle, fillcolor=255, center=(temp_size//2+5, temp_size//2+5))
                img = img.crop((5, 5, 5+temp_size, 5+temp_size))
            else:
                draw.text((x, y), letter, font=font, fill=0)

            # Random effects
            # Blur
            if random.random() < 0.15:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

            # Add noise
            if random.random() < 0.2:
                img_array = np.array(img, dtype=np.float32)
                noise = np.random.normal(0, random.uniform(3, 10), img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)

            # Random brightness/contrast
            if random.random() < 0.3:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.6, 1.5))
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.7, 1.4))

            # Add random background texture (lines, spots)
            if random.random() < 0.15:
                img_array = np.array(img, dtype=np.float32)
                # Add a faint line
                if random.random() < 0.5:
                    y_line = random.randint(0, temp_size - 1)
                    thickness = random.randint(1, 3)
                    for t in range(thickness):
                        if 0 <= y_line + t < temp_size:
                            img_array[y_line + t, :] = np.minimum(
                                img_array[y_line + t, :] + random.uniform(15, 40), 255
                            )
                img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            # Threshold to binarize (simulate scanned/printed look)
            if random.random() < 0.4:
                img_array = np.array(img, dtype=np.float32)
                threshold = random.uniform(100, 180)
                img_array = np.where(img_array < threshold, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)

            # Downscale to 28x28
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Invert if needed (char should be dark on light bg -> inverted for model: bright on dark)
            # Our handwritten dataset has dark bg (0), bright char (1). Let's match.
            img_array = 1.0 - img_array

            images.append(img_array)
            labels.append(letter_idx)

    return images, labels


# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SYNTHETIC PRINTED CHARACTER GENERATOR")
    print("=" * 60)

    fonts = find_fonts()
    if len(fonts) < 5:
        print("Not enough fonts found! Need at least 5.")
        sys.exit(1)

    # Select a diverse subset of ~20 fonts
    random.seed(42)
    selected_fonts = random.sample(fonts, min(25, len(fonts)))
    print(f"Using {len(selected_fonts)} fonts for generation")

    all_images = []
    all_labels = []
    samples_per_font = 80  # 80 * 20 fonts * 26 letters = 41,600 images

    for i, letter in enumerate(LETTERS):
        imgs, lbls = generate_for_letter(letter, i, selected_fonts, samples_per_font)
        all_images.extend(imgs)
        all_labels.extend(lbls)
        print(f"  {letter}: {len(imgs)} images generated")

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Save as numpy arrays (NHWC format for MLX)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    output_path = OUTPUT_DIR / "printed_letters.npz"
    np.savez_compressed(output_path, X=X, y=y)
    print(f"\nSaved {X.shape[0]} images to {output_path}")
    print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")


if __name__ == "__main__":
    main()
