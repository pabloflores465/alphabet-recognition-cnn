#!/usr/bin/env python3
"""
Real-time A-Z Alphabet Recognition Webcam Demo
Uses the robust CNN model trained on handwritten + printed characters.
MLX on Apple Silicon for fast inference.

Controls:
  Q / ESC  - Quit
  S        - Toggle between ROI mode and Auto-detect mode
  F        - Freeze/unfreeze frame
  C        - Clear frozen detection
  1-3      - Adjust threshold (auto mode)
  R        - Reset ROI position
"""

import cv2
import numpy as np
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ─── MODEL DEFINITIONS ───────────────────────────────────────────────
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]

class AlphabetCNN(nn.Module):
    """Original model: 4 conv blocks, 256->128->26 classifier."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )
    
    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = nn.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = nn.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = nn.relu(self.bn4(self.conv4(x))); x = self.pool4(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────
def load_model(model_path=None):
    if model_path is None:
        candidates = [
            Path(__file__).parent / "models" / "alphabet_cnn_mlx.safetensors",
            Path(__file__).parent / "models" / "alphabet_cnn_best.safetensors",
        ]
        for p in candidates:
            if p.exists():
                model_path = p
                break

    if model_path is None or not Path(model_path).exists():
        print("No trained model found! Run train_alphabet_cnn.py first.")
        for p in candidates:
            print(f"  {p}")
        return None

    model = AlphabetCNN()
    mx.eval(model.parameters())
    model.load_weights(str(model_path))
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"Accuracy: 99.2% (handwritten) - Preprocessing handles variations")
    return model


# ─── PREPROCESSING ─────────────────────────────────────────────────────
def preprocess_roi(roi):
    """Preprocess a region of interest for the CNN.
    Args:
        roi: numpy array (H, W, 3) BGR or (H, W) grayscale
    Returns:
        numpy array (1, 28, 28, 1) normalized NHWC
    """
    if roi.size == 0:
        return None

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Invert if light background (assume character is dark on light bg)
    # CNN expects bright char on dark bg (0-1 where 1 = character)
    mean_val = gray.mean()
    if mean_val > 128:
        gray = 255 - gray

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Threshold to binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours to center the character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Use largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Add padding
        pad = int(max(w, h) * 0.15)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(gray.shape[1] - x, w + 2 * pad)
        h = min(gray.shape[0] - y, h + 2 * pad)
        # Extract and resize
        char_img = binary[y:y+h, x:x+w]
        if char_img.size == 0:
            char_img = binary
    else:
        char_img = binary

    # Resize to 28x28 preserving aspect ratio
    h, w = char_img.shape
    if h == 0 or w == 0:
        return None

    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < 1: new_h = 1
    if new_w < 1: new_w = 1

    char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28 canvas
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_resized

    # Normalize and reshape to NHWC
    img_norm = canvas.astype(np.float32) / 255.0
    img_norm = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_norm, binary  # Return both for visualization


def auto_detect_characters(frame_bgr, min_area=500, max_area=50000):
    """Detect potential character regions in a frame.
    Returns list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter by aspect ratio (letters are roughly square-ish)
            aspect = w / h if h > 0 else 1
            if 0.1 < aspect < 5.0:
                # Add padding
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(frame_bgr.shape[1] - x, w + 2 * pad_x)
                h = min(frame_bgr.shape[0] - y, h + 2 * pad_y)
                boxes.append((x, y, w, h))

    return boxes, binary


# ─── PREDICTION ────────────────────────────────────────────────────────
def predict(model, img_norm):
    """Run inference on preprocessed image."""
    x = mx.array(img_norm)
    logits = model(x)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    return np.array(probs[0])


# ─── VISUALIZATION ─────────────────────────────────────────────────────
def draw_predictions(frame, probs, x, y, w, h, top_n=3):
    """Draw prediction bars and labels on frame."""
    top_indices = np.argsort(probs)[::-1][:top_n]

    bar_x = x + w + 10
    bar_max_width = 150
    bar_height = 22
    bar_spacing = 4

    for i, idx in enumerate(top_indices):
        conf = probs[idx] * 100
        letter = LETTERS[idx]
        bar_y = y + i * (bar_height + bar_spacing)

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height),
                      (60, 60, 60), -1)

        # Confidence bar
        bar_w = int(bar_max_width * conf / 100)
        if i == 0:
            color = (0, 255, 0) if conf > 60 else (0, 200, 255)
        elif i == 1:
            color = (255, 180, 0)
        else:
            color = (100, 100, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_height),
                      color, -1)

        # Text
        text = f"{letter}: {conf:.1f}%"
        cv2.putText(frame, text, (bar_x + 5, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Main prediction
    top_letter = LETTERS[top_indices[0]]
    top_conf = probs[top_indices[0]] * 100
    label = f"{top_letter} ({top_conf:.0f}%)"
    font_scale = min(2.0, max(1.0, w / 50))
    thickness = max(2, int(font_scale * 2))
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)


def draw_roi(frame, roi_x, roi_y, roi_size, color=(0, 255, 0)):
    """Draw ROI guide box."""
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), color, 2)
    # Crosshair
    cx, cy = roi_x + roi_size // 2, roi_y + roi_size // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 1)
    # Text prompt
    cv2.putText(frame, "Put letter here", (roi_x, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def draw_auto_boxes(frame, boxes, probs_list=None, threshold=60):
    """Draw detected boxes in auto mode."""
    if probs_list is None:
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 0), 2)
    else:
        for (x, y, w, h), probs in zip(boxes, probs_list):
            top_idx = np.argmax(probs)
            conf = probs[top_idx] * 100
            color = (0, 255, 0) if conf > threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{LETTERS[top_idx]} {conf:.0f}%"
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  A-Z ALPHABET RECOGNITION - LIVE WEBCAM DEMO")
    print("=" * 60)

    # Load model
    model = load_model()
    if model is None:
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam!")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Desactivar auto-ajustes de cámara
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)        # Auto-focus OFF
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto-exposure OFF (manual)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)          # Exposición fija
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)        # Auto-WB OFF
    except:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOCUS, 0)          # Focus fijo
    except:
        pass

    # Get actual frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {frame_w}x{frame_h}")

    # State
    mode = "roi"  # "roi" or "auto"
    roi_size = 200
    roi_x = (frame_w - roi_size) // 2
    roi_y = (frame_h - roi_size) // 2
    rotation = 0  # 0=normal, 1=mirror, 2=180°, 3=mirror+180°
    frozen = False
    frozen_frame = None
    frozen_probs = None
    auto_threshold = 60  # confidence threshold for auto mode

    print("\nControls:")
    print("  Q/ESC  - Quit")
    print("  S      - Switch ROI / Auto-detect mode")
    print("  F      - Freeze/unfreeze frame")
    print("  1-3    - Adjust confidence threshold")
    print("  Arrows - Move ROI")
    print("  +/-    - Resize ROI")
    print("  R      - Reset ROI")
    print("-" * 60)

    last_pred_time = 0
    pred_interval = 0.05  # 50ms between predictions (20 FPS)
    fps_counter = []
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply manual rotation (NOT automatic)
        if rotation == 0:
            pass  # normal
        elif rotation == 1:
            frame = cv2.flip(frame, 1)  # mirror horizontal
        elif rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 3:
            frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 1)

        if frozen and frozen_frame is not None:
            display_frame = frozen_frame.copy()
        else:
            display_frame = frame.copy()

        t_start = time.time()

        if mode == "roi" and not frozen:
            # Extract ROI
            r_x = max(0, roi_x)
            r_y = max(0, roi_y)
            r_s = min(roi_size, frame_w - r_x, frame_h - r_y)

            if r_s > 20:
                roi = frame[r_y:r_y+r_s, r_x:r_x+r_s]
                result = preprocess_roi(roi)
                if result is not None and time.time() - last_pred_time > pred_interval:
                    img_norm, binary = result
                    probs = predict(model, img_norm)
                    last_pred_time = time.time()

                    # Draw predictions
                    draw_predictions(display_frame, probs, r_x, r_y, r_s, r_s)
                    # Show binary preview in corner
                    preview_h, preview_w = binary.shape
                    preview_scale = 120 / max(preview_h, preview_w)
                    preview_resized = cv2.resize(binary, None, fx=preview_scale, fy=preview_scale)
                    preview_color = cv2.cvtColor(preview_resized, cv2.COLOR_GRAY2BGR)
                    ph, pw = preview_color.shape[:2]
                    display_frame[10:10+ph, frame_w-pw-10:frame_w-10] = preview_color

            # Draw ROI
            draw_roi(display_frame, roi_x, roi_y, roi_size)

        elif mode == "auto" and not frozen:
            boxes, binary = auto_detect_characters(frame, min_area=300, max_area=80000)

            if boxes and time.time() - last_pred_time > pred_interval:
                probs_list = []
                valid_boxes = []
                for (x, y, w, h) in boxes:
                    roi = frame[y:y+h, x:x+w]
                    result = preprocess_roi(roi)
                    if result is not None:
                        img_norm, _ = result
                        probs = predict(model, img_norm)
                        probs_list.append(probs)
                        valid_boxes.append((x, y, w, h))
                last_pred_time = time.time()
                draw_auto_boxes(display_frame, valid_boxes, probs_list, auto_threshold)

            # Show binary debug in corner
            preview_scale = 200 / max(binary.shape[0], binary.shape[1])
            preview_resized = cv2.resize(binary, None, fx=preview_scale, fy=preview_scale)
            preview_color = cv2.cvtColor(preview_resized, cv2.COLOR_GRAY2BGR)
            ph, pw = preview_color.shape[:2]
            display_frame[10:10+ph, frame_w-pw-10:frame_w-10] = preview_color
            draw_auto_boxes(display_frame, boxes)

        # FPS counter
        fps_counter.append(time.time())
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        if len(fps_counter) >= 2:
            fps_display = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])

        # Status bar
        rot_names = ['NORMAL', 'MIRROR', '180°', 'MIRROR+180°']
        mode_text = f"MODE: {'ROI' if mode == 'roi' else 'AUTO'} | ROT: {rot_names[rotation]} | FPS: {fps_display:.1f}"
        if mode == 'auto':
            mode_text += f" | THRESH: {auto_threshold}%"
        if frozen:
            mode_text += " | FROZEN"

        cv2.putText(display_frame, mode_text, (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Alphabet Recognition - Webcam Demo", display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('s'):
            mode = "auto" if mode == "roi" else "roi"
            print(f"Mode: {mode}")
        elif key == ord('f'):
            if not frozen:
                frozen = True
                frozen_frame = frame.copy()
                if mode == "roi":
                    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                    result = preprocess_roi(roi)
                    if result is not None:
                        img_norm, _ = result
                        frozen_probs = predict(model, img_norm)
                print("Frame frozen. Press F to unfreeze.")
            else:
                frozen = False
                frozen_frame = None
                frozen_probs = None
                print("Unfrozen.")
        elif key == ord('r'):
            roi_x = (frame_w - roi_size) // 2
            roi_y = (frame_h - roi_size) // 2
        elif key == ord('m'):  # Manual rotation
            rotation = (rotation + 1) % 4
            names = ['Normal', 'Mirror', '180', 'Mirror+180']
            print(f"Rotation: {names[rotation]}")
        elif key == ord('1'):
            auto_threshold = 40
        elif key == ord('2'):
            auto_threshold = 60
        elif key == ord('3'):
            auto_threshold = 80
        elif key == ord('=') or key == ord('+'):
            roi_size = min(400, roi_size + 10)
            roi_x = (frame_w - roi_size) // 2
            roi_y = (frame_h - roi_size) // 2
        elif key == ord('-') or key == ord('_'):
            roi_size = max(60, roi_size - 10)
            roi_x = (frame_w - roi_size) // 2
            roi_y = (frame_h - roi_size) // 2
        # Arrow keys
        elif key == 81:  # left
            roi_x = max(0, roi_x - 10)
        elif key == 82:  # up
            roi_y = max(0, roi_y - 10)
        elif key == 83:  # right
            roi_x = min(frame_w - roi_size, roi_x + 10)
        elif key == 84:  # down
            roi_y = min(frame_h - roi_size, roi_y + 10)

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
