#!/usr/bin/env python3
"""
A-Z Alphabet Recognition — GUI Application
Tkinter-based GUI with webcam feed, real-time predictions, and controls.
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import time
import threading
from pathlib import Path
from collections import deque

import mlx.core as mx
import mlx.nn as nn

# ─── MODEL ─────────────────────────────────────────────────────────────
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]

class AlphabetCNN(nn.Module):
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


# ─── PREPROCESSING ─────────────────────────────────────────────────────
def preprocess_roi(roi):
    if roi.size == 0:
        return None

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    mean_val = gray.mean()
    if mean_val > 128:
        gray = 255 - gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(max(w, h) * 0.15)
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2 * pad)
        h = min(binary.shape[0] - y, h + 2 * pad)
        char_img = binary[y:y+h, x:x+w]
    else:
        char_img = binary

    if char_img.size == 0:
        return None

    h, w = char_img.shape
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_off = (IMG_SIZE - new_h) // 2
    x_off = (IMG_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_resized

    img_norm = canvas.astype(np.float32) / 255.0
    return img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1), binary


def predict(model, img_norm):
    x = mx.array(img_norm)
    logits = model(x)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    return np.array(probs[0])


# ─── GUI APPLICATION ───────────────────────────────────────────────────
class AlphabetRecognitionApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("A-Z Alphabet Recognition")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Style
        self.bg_color = "#1a1a2e"
        self.card_color = "#16213e"
        self.accent_color = "#0f3460"
        self.highlight = "#e94560"
        self.text_color = "#ffffff"
        self.text_secondary = "#a0a0b0"
        self.success_color = "#00cec9"

        # State
        self.running = True
        self.roi_size = 200
        self.roi_x = 0
        self.roi_y = 0
        self.rotation = 0  # 0=normal, 1=mirror, 2=180°, 3=mirror+180°
        self.current_probs = None
        self.prediction_history = deque(maxlen=20)
        self.fps = 0
        self.fps_counter = deque(maxlen=30)

        # Setup
        self.setup_camera()
        self.build_ui()

        # Start update loop
        self.update_frame()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Desactivar auto-ajustes de cámara
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)        # Auto-focus OFF
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto-exposure OFF (manual)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)          # Exposición fija
        # Auto white balance (si existe la propiedad)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)        # Auto-WB OFF
        except:
            pass
        # Fijar focus manual (si se soporta)
        try:
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)          # Focus fijo
        except:
            pass

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi_x = (self.frame_w - self.roi_size) // 2
        self.roi_y = (self.frame_h - self.roi_size) // 2

    def build_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ─── LEFT PANEL: Webcam ───
        self.left_panel = tk.Frame(self.main_frame, bg=self.card_color,
                                   highlightbackground=self.accent_color,
                                   highlightthickness=1)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Title
        self.cam_title = tk.Label(self.left_panel, text="📷 Webcam Feed",
                                  font=("Helvetica", 13, "bold"),
                                  fg=self.text_color, bg=self.card_color)
        self.cam_title.pack(pady=(8, 4))

        # Webcam canvas
        self.cam_canvas = tk.Canvas(self.left_panel, bg="black",
                                    width=640, height=480,
                                    highlightthickness=0)
        self.cam_canvas.pack(padx=10, pady=(0, 10))

        # FPS label
        self.fps_label = tk.Label(self.left_panel, text="FPS: --",
                                  font=("Helvetica", 9),
                                  fg=self.text_secondary, bg=self.card_color)
        self.fps_label.pack(pady=(0, 8))

        # ─── RIGHT PANEL: Predictions ───
        self.right_panel = tk.Frame(self.main_frame, bg=self.card_color,
                                    highlightbackground=self.accent_color,
                                    highlightthickness=1, width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        self.right_panel.pack_propagate(False)

        # Title
        pred_title = tk.Label(self.right_panel, text="🧠 Predictions",
                              font=("Helvetica", 14, "bold"),
                              fg=self.success_color, bg=self.card_color)
        pred_title.pack(pady=(12, 8))

        # Main prediction (big)
        self.main_pred_frame = tk.Frame(self.right_panel, bg=self.card_color)
        self.main_pred_frame.pack(pady=(5, 10))

        self.main_letter_label = tk.Label(self.main_pred_frame, text="--",
                                          font=("Helvetica", 72, "bold"),
                                          fg=self.highlight, bg=self.card_color)
        self.main_letter_label.pack()

        self.main_conf_label = tk.Label(self.main_pred_frame, text="--%",
                                        font=("Helvetica", 18),
                                        fg=self.text_secondary, bg=self.card_color)
        self.main_conf_label.pack()

        # Separator
        sep = tk.Frame(self.right_panel, bg=self.accent_color, height=1)
        sep.pack(fill=tk.X, padx=20, pady=(5, 10))

        # Top-3 bars
        self.bar_frame = tk.Frame(self.right_panel, bg=self.card_color)
        self.bar_frame.pack(fill=tk.X, padx=20, pady=(0, 5))

        self.bar_labels = []
        self.bar_canvases = []
        LETTER_COLORS = ["#00cec9", "#fdcb6e", "#e17055"]
        for i in range(3):
            row = tk.Frame(self.bar_frame, bg=self.card_color)
            row.pack(fill=tk.X, pady=2)

            letter_lbl = tk.Label(row, text="--", font=("Helvetica", 20, "bold"),
                                  fg=LETTER_COLORS[i], bg=self.card_color, width=3)
            letter_lbl.pack(side=tk.LEFT, padx=(0, 8))

            canvas = tk.Canvas(row, bg="#0a0a1a", height=22,
                               highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

            pct_lbl = tk.Label(row, text="--%", font=("Helvetica", 11, "bold"),
                               fg=self.text_color, bg=self.card_color, width=6)
            pct_lbl.pack(side=tk.RIGHT, padx=(8, 0))

            self.bar_labels.append((letter_lbl, pct_lbl))
            self.bar_canvases.append((canvas, LETTER_COLORS[i]))

        # Separator
        sep2 = tk.Frame(self.right_panel, bg=self.accent_color, height=1)
        sep2.pack(fill=tk.X, padx=20, pady=(10, 5))

        # History
        hist_title = tk.Label(self.right_panel, text="📋 History",
                              font=("Helvetica", 11, "bold"),
                              fg=self.text_color, bg=self.card_color)
        hist_title.pack(pady=(5, 4))

        self.history_frame = tk.Frame(self.right_panel, bg=self.card_color)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        self.history_text = tk.Text(self.history_frame, bg="#0a0a1a",
                                    fg=self.text_color, font=("Courier", 10),
                                    height=10, width=28, borderwidth=0,
                                    state=tk.DISABLED, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # ─── BOTTOM BAR: Controls ───
        self.control_bar = tk.Frame(self.root, bg=self.accent_color, height=45)
        self.control_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.control_bar.pack_propagate(False)

        # Control buttons
        btn_style = {"font": ("Helvetica", 10, "bold"), "bg": self.card_color,
                     "fg": self.text_color, "activebackground": self.highlight,
                     "activeforeground": "white", "bd": 0, "padx": 15, "pady": 6,
                     "cursor": "hand2"}

        self.btn_freeze = tk.Button(self.control_bar, text="⏸ Freeze",
                                    command=self.toggle_freeze, **btn_style)
        self.btn_freeze.pack(side=tk.LEFT, padx=(10, 5), pady=6)

        self.btn_roi_bigger = tk.Button(self.control_bar, text="➕ ROI Bigger",
                                        command=self.roi_bigger, **btn_style)
        self.btn_roi_bigger.pack(side=tk.LEFT, padx=5, pady=6)

        self.btn_roi_smaller = tk.Button(self.control_bar, text="➖ ROI Smaller",
                                         command=self.roi_smaller, **btn_style)
        self.btn_roi_smaller.pack(side=tk.LEFT, padx=5, pady=6)

        self.btn_reset = tk.Button(self.control_bar, text="🔄 Reset ROI",
                                   command=self.reset_roi, **btn_style)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=6)

        self.btn_rotate = tk.Button(self.control_bar, text="🔃 Rotate",
                                     command=self.cycle_rotation, **btn_style)
        self.btn_rotate.pack(side=tk.LEFT, padx=5, pady=6)

        # Quit button (right side)
        self.btn_quit = tk.Button(self.control_bar, text="✕ Quit",
                                  command=self.on_closing,
                                  font=("Helvetica", 10, "bold"),
                                  bg=self.highlight, fg="white",
                                  activebackground="#ff6b81",
                                  bd=0, padx=15, pady=6, cursor="hand2")
        self.btn_quit.pack(side=tk.RIGHT, padx=10, pady=6)

        # Keyboard bindings
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('q', lambda e: self.on_closing())
        self.root.bind('f', lambda e: self.toggle_freeze())
        self.root.bind('r', lambda e: self.reset_roi())
        self.root.bind('<Left>', lambda e: self.move_roi(-10, 0))
        self.root.bind('<Right>', lambda e: self.move_roi(10, 0))
        self.root.bind('<Up>', lambda e: self.move_roi(0, -10))
        self.root.bind('<Down>', lambda e: self.move_roi(0, 10))
        self.root.bind('<plus>', lambda e: self.roi_bigger())
        self.root.bind('<minus>', lambda e: self.roi_smaller())
        self.root.bind('<equal>', lambda e: self.roi_bigger())
        self.root.bind('m', lambda e: self.cycle_rotation())

    def cycle_rotation(self):
        self.rotation = (self.rotation + 1) % 4
        names = ['Normal', 'Mirror', '180°', 'Mirror+180°']
        self.btn_rotate.config(text=f'🔃 {names[self.rotation]}')

    def toggle_freeze(self):
        if hasattr(self, 'frozen'):
            self.frozen = not self.frozen
            self.btn_freeze.config(
                text="▶ Unfreeze" if self.frozen else "⏸ Freeze",
                bg=self.highlight if self.frozen else self.card_color)
        else:
            self.frozen = True
            self.frozen_frame = None
            self.btn_freeze.config(text="▶ Unfreeze", bg=self.highlight)

    def roi_bigger(self):
        self.roi_size = min(400, self.roi_size + 15)
        self.roi_x = (self.frame_w - self.roi_size) // 2
        self.roi_y = (self.frame_h - self.roi_size) // 2

    def roi_smaller(self):
        self.roi_size = max(60, self.roi_size - 15)
        self.roi_x = (self.frame_w - self.roi_size) // 2
        self.roi_y = (self.frame_h - self.roi_size) // 2

    def reset_roi(self):
        self.roi_size = 200
        self.roi_x = (self.frame_w - self.roi_size) // 2
        self.roi_y = (self.frame_h - self.roi_size) // 2

    def move_roi(self, dx, dy):
        self.roi_x = max(0, min(self.frame_w - self.roi_size, self.roi_x + dx))
        self.roi_y = max(0, min(self.frame_h - self.roi_size, self.roi_y + dy))

    def update_predictions_display(self, probs):
        top_indices = np.argsort(probs)[::-1][:3]

        # Main prediction
        top_letter = LETTERS[top_indices[0]]
        top_conf = probs[top_indices[0]] * 100
        self.main_letter_label.config(text=top_letter)
        self.main_conf_label.config(text=f"{top_conf:.1f}%")

        # Color main letter based on confidence
        if top_conf > 80:
            self.main_letter_label.config(fg="#00cec9")
        elif top_conf > 50:
            self.main_letter_label.config(fg="#fdcb6e")
        else:
            self.main_letter_label.config(fg="#e17055")

        # Top-3 bars
        for i, idx in enumerate(top_indices):
            letter, pct = self.bar_labels[i]
            canvas, color = self.bar_canvases[i]

            letter.config(text=LETTERS[idx])
            pct.config(text=f"{probs[idx]*100:.1f}%")

            # Redraw bar
            canvas.delete("all")
            bar_width = int((canvas.winfo_width() if canvas.winfo_width() > 1 else 200)
                            * probs[idx])
            canvas.create_rectangle(0, 0, bar_width, 24, fill=color, outline="")

    def add_to_history(self, probs):
        if probs is None:
            return

        top_idx = np.argmax(probs)
        top_letter = LETTERS[top_idx]
        top_conf = probs[top_idx] * 100

        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp}  {top_letter}  {top_conf:5.1f}%"

        self.prediction_history.append(entry)

        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        for e in reversed(self.prediction_history):
            self.history_text.insert(tk.END, e + "\n")
        self.history_text.config(state=tk.DISABLED)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        t0 = time.time()
        
        # Manual rotation (not automatic)
        if self.rotation == 0:
            pass  # normal
        elif self.rotation == 1:
            frame = cv2.flip(frame, 1)  # mirror horizontal
        elif self.rotation == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 3:
            frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 1)
        
        display = frame.copy()

        # Frozen state
        if hasattr(self, 'frozen') and self.frozen:
            if self.frozen_frame is None:
                self.frozen_frame = frame.copy()
                # Process frozen frame
                self.process_and_predict(self.frozen_frame, display_too=False)
            display = self.frozen_frame.copy()
        else:
            self.frozen_frame = None
            self.process_and_predict(frame)

        # Draw ROI box on display
        rx, ry, rs = self.roi_x, self.roi_y, self.roi_size
        cv2.rectangle(display, (rx, ry), (rx + rs, ry + rs), (0, 206, 201), 2)
        cv2.rectangle(display, (rx-1, ry-1), (rx+rs+1, ry+rs+1), (0, 0, 0), 1)
        # Crosshair
        cx, cy = rx + rs // 2, ry + rs // 2
        cv2.line(display, (cx-15, cy), (cx+15, cy), (0, 206, 201), 1)
        cv2.line(display, (cx, cy-15), (cx, cy+15), (0, 206, 201), 1)

        # ROI size indicator
        cv2.putText(display, f"ROI: {rs}px", (rx, ry - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 206, 201), 1)

        # Convert to PIL ImageTk format
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(display_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.cam_canvas.create_image(
            self.cam_canvas.winfo_width() // 2,
            self.cam_canvas.winfo_height() // 2,
            image=img_tk, anchor=tk.CENTER
        )
        self.cam_canvas.image = img_tk  # Keep reference

        # FPS
        self.fps_counter.append(time.time())
        if len(self.fps_counter) > 2:
            fps = (len(self.fps_counter) - 1) / \
                  (self.fps_counter[-1] - self.fps_counter[0])
            self.fps_label.config(text=f"FPS: {fps:.1f}")

        self.root.after(10, self.update_frame)

    def process_and_predict(self, frame, display_too=True):
        rx = max(0, self.roi_x)
        ry = max(0, self.roi_y)
        rs = min(self.roi_size, self.frame_w - rx, self.frame_h - ry)

        if rs <= 20:
            return

        roi = frame[ry:ry+rs, rx:rx+rs]
        result = preprocess_roi(roi)
        if result is not None:
            img_norm, _ = result
            probs = predict(self.model, img_norm)
            self.current_probs = probs

            if display_too:
                self.update_predictions_display(probs)

            # Add to history if confidence changed significantly
            if len(self.prediction_history) == 0 or \
               LETTERS[np.argmax(probs)] != self.prediction_history[-1].split()[2] or \
               abs(probs[np.argmax(probs)] - float(self.prediction_history[-1].split()[-1].replace('%', '')) / 100) > 0.05:
                self.add_to_history(probs)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def run(self):
        # Center window
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 800: w = 980
        if h < 520: h = 580
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.root.mainloop()


# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    # Load model
    model_dir = Path(__file__).parent / "models"
    candidates = [
        model_dir / "alphabet_cnn_mlx.safetensors",
        model_dir / "alphabet_cnn_best.safetensors",
    ]
    model_path = None
    for p in candidates:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        print("No model found! Run train_alphabet_cnn.py first.")
        return

    model = AlphabetCNN()
    mx.eval(model.parameters())
    model.load_weights(str(model_path))
    model.eval()
    print(f"Model loaded: {model_path.name} (99.2% accuracy)")

    # Launch GUI
    app = AlphabetRecognitionApp(model)
    app.run()


if __name__ == "__main__":
    main()
