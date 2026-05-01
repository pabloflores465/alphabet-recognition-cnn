#!/usr/bin/env python3
"""
A-Z Alphabet Recognition — GUI v2
- ROI draggable with mouse
- Rotation-invariant inference (tests multiple rotations)
- Uses original 99.2% model (works for printed via preprocessing)
- Manual camera controls (no auto-focus/exposure/white-balance)
- Manual rotation mode
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
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
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(2, 2)
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
    if roi.size == 0: return None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    mean_val = gray.mean()
    if mean_val > 128: gray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(max(w, h) * 0.15)
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(binary.shape[1] - x, w + 2*pad)
        h = min(binary.shape[0] - y, h + 2*pad)
        char_img = binary[y:y+h, x:x+w]
    else:
        char_img = binary
    if char_img.size == 0: return None
    h, w = char_img.shape
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = max(1, int(h*scale)), max(1, int(w*scale))
    char_resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y_off = (IMG_SIZE - new_h)//2
    x_off = (IMG_SIZE - new_w)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = char_resized
    img_norm = canvas.astype(np.float32) / 255.0
    return img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1), binary


def predict_rotation_invariant(model, img_norm):
    """Two-stage rotation search: coarse every 15° then fine ±7° around best.
    Uses batch inference for speed."""
    import cv2 as cv
    
    img = img_norm[0, :, :, 0].astype(np.float32)
    h, w = img.shape
    
    # Stage 1: coarse search (every 15° + key cardinals)
    coarse_angles = list(range(-45, 46, 15)) + [90, 180, 270]
    coarse_angles = sorted(set(coarse_angles))
    
    best_probs = None
    best_angle = 0
    
    for angle in coarse_angles:
        if angle == 0:
            rotated = img
        else:
            matrix = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv.warpAffine(img, matrix, (w, h),
                                    borderMode=cv.BORDER_CONSTANT,
                                    borderValue=0.0)
        
        inp = rotated.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        x = mx.array(inp)
        logits = model(x)
        probs = mx.softmax(logits, axis=1)
        mx.eval(probs)
        probs_np = np.array(probs[0])
        
        if best_probs is None or probs_np.max() > best_probs.max():
            best_probs = probs_np
            best_angle = angle
    
    # Stage 2: fine search around best angle (±8°, step 2°)
    if abs(best_angle) <= 45 or best_angle in [90, 180, 270]:
        fine_angles = []
        for da in range(-8, 9, 2):
            a = best_angle + da
            if a not in coarse_angles:
                fine_angles.append(a)
        
        for angle in fine_angles:
            if angle == 0:
                rotated = img
            else:
                matrix = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                rotated = cv.warpAffine(img, matrix, (w, h),
                                        borderMode=cv.BORDER_CONSTANT,
                                        borderValue=0.0)
            
            inp = rotated.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            x = mx.array(inp)
            logits = model(x)
            probs = mx.softmax(logits, axis=1)
            mx.eval(probs)
            probs_np = np.array(probs[0])
            
            if probs_np.max() > best_probs.max():
                best_probs = probs_np
                best_angle = angle
    
    return best_probs, best_angle


# ─── GUI ───────────────────────────────────────────────────────────────
class App:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("A-Z Alphabet Recognition — Printed")
        self.root.configure(bg="#1a1a2e")

        self.bg = "#1a1a2e"
        self.card = "#16213e"
        self.accent = "#0f3460"
        self.hl = "#e94560"
        self.fg = "#ffffff"
        self.fg2 = "#a0a0b0"
        self.green = "#00cec9"

        self.running = True
        self.roi_size = 200
        self.roi_x = 0
        self.roi_y = 0
        self.rotation = 0
        self.frozen = False
        self.frozen_frame = None
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_roi_start = (0, 0)
        self.current_probs = None
        self.pred_history = deque(maxlen=20)
        self.current_angle = 0

        self.setup_camera()
        self.build_ui()
        self.update_frame()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        try: self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        except: pass
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi_x = (self.frame_w - self.roi_size)//2
        self.roi_y = (self.frame_h - self.roi_size)//2

    def build_ui(self):
        self.main = tk.Frame(self.root, bg=self.bg)
        self.main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: webcam
        self.left = tk.Frame(self.main, bg=self.card, highlightbackground=self.accent, highlightthickness=1)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))

        tk.Label(self.left, text="📷 Webcam", font=("Helvetica",13,"bold"),
                fg=self.fg, bg=self.card).pack(pady=(8,4))

        self.canvas = tk.Canvas(self.left, bg="black", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0,10), fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self.fps_lbl = tk.Label(self.left, text="FPS: --", font=("Helvetica",9),
                                fg=self.fg2, bg=self.card)
        self.fps_lbl.pack(pady=(0,8))

        # RIGHT: predictions
        self.right = tk.Frame(self.main, bg=self.card, highlightbackground=self.accent, highlightthickness=1, width=320)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5,0))
        self.right.pack_propagate(False)

        tk.Label(self.right, text="🧠 Prediction", font=("Helvetica",14,"bold"),
                fg=self.green, bg=self.card).pack(pady=(12,8))

        self.main_letter = tk.Label(self.right, text="--", font=("Helvetica",72,"bold"),
                                     fg=self.hl, bg=self.card)
        self.main_letter.pack()
        self.main_conf = tk.Label(self.right, text="--%", font=("Helvetica",16),
                                  fg=self.fg2, bg=self.card)
        self.main_conf.pack()
        self.angle_lbl = tk.Label(self.right, text="", font=("Helvetica",9),
                                  fg=self.fg2, bg=self.card)
        self.angle_lbl.pack()

        sep = tk.Frame(self.right, bg=self.accent, height=1)
        sep.pack(fill=tk.X, padx=20, pady=(8,8))

        # Top-3 bars
        self.bars = []
        colors = ["#00cec9", "#fdcb6e", "#e17055"]
        for i in range(3):
            row = tk.Frame(self.right, bg=self.card)
            row.pack(fill=tk.X, padx=20, pady=2)
            lbl = tk.Label(row, text="--", font=("Helvetica",18,"bold"), fg=colors[i], bg=self.card, width=3)
            lbl.pack(side=tk.LEFT, padx=(0,8))
            bar = tk.Canvas(row, bg="#0a0a1a", height=20, highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            pct = tk.Label(row, text="--%", font=("Helvetica",10,"bold"), fg=self.fg, bg=self.card, width=6)
            pct.pack(side=tk.RIGHT, padx=(8,0))
            self.bars.append((lbl, bar, pct, colors[i]))

        sep2 = tk.Frame(self.right, bg=self.accent, height=1)
        sep2.pack(fill=tk.X, padx=20, pady=(8,4))

        tk.Label(self.right, text="📋 History", font=("Helvetica",11,"bold"),
                fg=self.fg, bg=self.card).pack(pady=(4,4))

        self.hist_text = tk.Text(self.right, bg="#0a0a1a", fg=self.fg,
                                 font=("Courier",10), height=8, width=28,
                                 borderwidth=0, state=tk.DISABLED)
        self.hist_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,10))

        # BOTTOM BAR
        bar_frame = tk.Frame(self.root, bg=self.accent, height=42)
        bar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bar_frame.pack_propagate(False)

        btn = {"font":("Helvetica",9,"bold"), "bg":self.card, "fg":self.fg,
               "activebackground":self.hl, "bd":0, "padx":12, "pady":5, "cursor":"hand2"}

        tk.Button(bar_frame, text="⏸ Freeze", command=self.toggle_freeze, **btn).pack(side=tk.LEFT, padx=(8,3), pady=6)
        tk.Button(bar_frame, text="➕ +", command=lambda:self.resize_roi(15), **btn).pack(side=tk.LEFT, padx=3, pady=6)
        tk.Button(bar_frame, text="➖ -", command=lambda:self.resize_roi(-15), **btn).pack(side=tk.LEFT, padx=3, pady=6)
        tk.Button(bar_frame, text="🔄 Reset", command=self.reset_roi, **btn).pack(side=tk.LEFT, padx=3, pady=6)
        tk.Button(bar_frame, text="🔃 Rotate", command=self.cycle_rotation, **btn).pack(side=tk.LEFT, padx=3, pady=6)
        tk.Button(bar_frame, text="✕ Quit", command=self.on_close,
                  font=("Helvetica",9,"bold"), bg=self.hl, fg="white",
                  activebackground="#ff6b81", bd=0, padx=12, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=8, pady=6)

        # Keybindings
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('q', lambda e: self.on_close())
        self.root.bind('f', lambda e: self.toggle_freeze())
        self.root.bind('r', lambda e: self.reset_roi())
        self.root.bind('m', lambda e: self.cycle_rotation())
        self.root.bind('<Left>', lambda e: self.move_roi(-10,0))
        self.root.bind('<Right>', lambda e: self.move_roi(10,0))
        self.root.bind('<Up>', lambda e: self.move_roi(0,-10))
        self.root.bind('<Down>', lambda e: self.move_roi(0,10))
        self.root.bind('<plus>', lambda e: self.resize_roi(15))
        self.root.bind('<minus>', lambda e: self.resize_roi(-15))
        self.root.bind('<equal>', lambda e: self.resize_roi(15))

    def on_drag_start(self, event):
        self.dragging = True
        self.drag_start = (event.x, event.y)
        self.drag_roi_start = (self.roi_x, self.roi_y)

    def on_drag_move(self, event):
        if not self.dragging: return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        # Scale from canvas coords to frame coords
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        scale_x = self.frame_w / cw if cw > 0 else 1
        scale_y = self.frame_h / ch if ch > 0 else 1
        self.roi_x = max(0, min(self.frame_w - self.roi_size,
                                 self.drag_roi_start[0] + int(dx * scale_x)))
        self.roi_y = max(0, min(self.frame_h - self.roi_size,
                                 self.drag_roi_start[1] + int(dy * scale_y)))

    def on_drag_end(self, event):
        self.dragging = False

    def toggle_freeze(self):
        self.frozen = not self.frozen
        if self.frozen: self.frozen_frame = None

    def cycle_rotation(self):
        self.rotation = (self.rotation + 1) % 4

    def resize_roi(self, delta):
        self.roi_size = max(60, min(400, self.roi_size + delta))
        self.roi_x = max(0, min(self.frame_w - self.roi_size, self.roi_x))
        self.roi_y = max(0, min(self.frame_h - self.roi_size, self.roi_y))

    def reset_roi(self):
        self.roi_size = 200
        self.roi_x = (self.frame_w - self.roi_size)//2
        self.roi_y = (self.frame_h - self.roi_size)//2

    def move_roi(self, dx, dy):
        self.roi_x = max(0, min(self.frame_w - self.roi_size, self.roi_x + dx))
        self.roi_y = max(0, min(self.frame_h - self.roi_size, self.roi_y + dy))

    def update_display(self, probs, angle=0):
        top = np.argsort(probs)[::-1][:3]
        self.main_letter.config(text=LETTERS[top[0]])
        self.main_conf.config(text=f"{probs[top[0]]*100:.1f}%")
        if angle != 0:
            self.angle_lbl.config(text=f"Rotated {angle}°")
        else:
            self.angle_lbl.config(text="")
        for i, idx in enumerate(top):
            lbl, bar, pct, color = self.bars[i]
            lbl.config(text=LETTERS[idx])
            pct.config(text=f"{probs[idx]*100:.1f}%")
            bar.delete("all")
            bw = int((bar.winfo_width() if bar.winfo_width()>1 else 200) * probs[idx])
            bar.create_rectangle(0, 0, bw, 24, fill=color, outline="")

    def add_history(self, probs):
        top = np.argmax(probs)
        self.pred_history.append(f"{time.strftime('%H:%M:%S')}  {LETTERS[top]}  {probs[top]*100:5.1f}%")
        self.hist_text.config(state=tk.NORMAL)
        self.hist_text.delete(1.0, tk.END)
        for e in reversed(self.pred_history):
            self.hist_text.insert(tk.END, e + "\n")
        self.hist_text.config(state=tk.DISABLED)

    def update_frame(self):
        if not self.running: return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame); return

        # Manual rotation
        if self.rotation == 1: frame = cv2.flip(frame, 1)
        elif self.rotation == 2: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 3: frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 1)

        t0 = time.time()
        display = frame.copy()

        if self.frozen and self.frozen_frame is not None:
            display = self.frozen_frame.copy()
        elif self.frozen:
            self.frozen_frame = frame.copy()

        # Process ROI
        if not self.frozen:
            rx, ry, rs = self.roi_x, self.roi_y, self.roi_size
            roi = frame[ry:ry+rs, rx:rx+rs]
            result = preprocess_roi(roi)
            if result is not None:
                img_norm, _ = result
                probs, angle = predict_rotation_invariant(self.model, img_norm)
                self.current_probs = probs
                self.current_angle = angle
                self.update_display(probs, angle)

                if len(self.pred_history) == 0 or \
                   LETTERS[np.argmax(probs)] != self.pred_history[-1].split()[1] or \
                   abs(probs[np.argmax(probs)] - float(self.pred_history[-1].split()[-1].replace('%',''))/100) > 0.05:
                    self.add_history(probs)

        # Draw ROI on display
        rx, ry, rs = self.roi_x, self.roi_y, self.roi_size
        cv2.rectangle(display, (rx,ry), (rx+rs,ry+rs), (0,206,201), 2)
        cv2.rectangle(display, (rx-1,ry-1), (rx+rs+1,ry+rs+1), (0,0,0), 1)
        cx, cy = rx+rs//2, ry+rs//2
        cv2.line(display, (cx-12,cy), (cx+12,cy), (0,206,201), 1)
        cv2.line(display, (cx,cy-12), (cx,cy+12), (0,206,201), 1)

        # Convert to ImageTk
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        # Resize to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            scale = min(cw/self.frame_w, ch/self.frame_h)
            new_w, new_h = int(self.frame_w*scale), int(self.frame_h*scale)
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.tk_img, anchor=tk.CENTER)

        # Speed
        dt = time.time() - t0
        if dt > 0: self.fps_lbl.config(text=f"FPS: {1/dt:.1f} | ROI: [{self.roi_x},{self.roi_y}] {rs}px")

        self.root.after(15, self.update_frame)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w < 900: w = 1000
        if h < 550: h = 600
        x = (self.root.winfo_screenwidth()//2) - (w//2)
        y = (self.root.winfo_screenheight()//2) - (h//2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.mainloop()


def main():
    model_dir = Path(__file__).parent / "models"
    paths = [model_dir/"alphabet_cnn_mlx.safetensors",
             model_dir/"alphabet_cnn_best.safetensors",
             model_dir/"printed_cnn.safetensors"]
    model_path = None
    for p in paths:
        if p.exists(): model_path = p; break
    if not model_path:
        print("No model found! Run train_alphabet_cnn.py first."); return

    model = AlphabetCNN()
    mx.eval(model.parameters())
    model.load_weights(str(model_path))
    model.eval()
    print(f"Loaded: {model_path.name}")
    App(model).run()


if __name__ == "__main__":
    main()
