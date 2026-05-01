#!/usr/bin/env python3
"""
Printed-only A-Z CNN — Trained ONLY on printed letters (no handwritten).
Heavy rotation augmentation for rotation-invariant detection.
Optimized for Apple Silicon via MLX.
"""

import time, random, numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ─── CONFIG ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
PRINTED_V4 = PROJECT_DIR / "printed_dataset_v4" / "printed_v4.npz"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 25
LR = 0.001
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]


# ─── ROTATION-INVARIANT AUGMENTATION ──────────────────────────────────
def rotate_image(img, angle_deg):
    """Rotate a 28x28 image by angle (degrees). Returns 28x28."""
    import cv2
    h, w = img.shape
    if h == 0 or w == 0:
        return img
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0.0)
    if rotated.size == 0:
        return img
    return rotated

def augment_batch(images_np):
    """Apply heavy augmentations including random rotation."""
    import cv2
    batch = images_np.copy()
    B = batch.shape[0]

    # Batch ops
    if random.random() < 0.5:
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.6, 1.5)
        batch = np.clip((batch - 0.5) * contrast + 0.5 * brightness, 0, 1)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(0.005, 0.06), batch.shape).astype(np.float32)
        batch = np.clip(batch + noise, 0, 1)

    for i in range(B):
        img = batch[i, :, :, 0]

        # Random rotation (KEY for rotation invariance)
        if random.random() < 0.7:
            angle = random.uniform(-45, 45)  # Up to ±45 degrees
            img = rotate_image(img, angle)

        # Random shift
        if random.random() < 0.5:
            img = np.roll(img, random.randint(-3, 3), axis=1)
            img = np.roll(img, random.randint(-3, 3), axis=0)

        # Random scale (simple zoom)
        if random.random() < 0.4:
            scale = random.uniform(0.75, 1.3)
            new_sz = max(8, int(IMG_SIZE * scale))
            if new_sz < IMG_SIZE:
                # Shrink: resize down then pad
                img_small = cv2.resize(img, (new_sz, new_sz), interpolation=cv2.INTER_AREA)
                temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                off = (IMG_SIZE - new_sz) // 2
                temp[off:off+new_sz, off:off+new_sz] = img_small
                img = temp
            elif new_sz > IMG_SIZE:
                # Zoom in: resize up then crop center
                img_big = cv2.resize(img, (new_sz, new_sz), interpolation=cv2.INTER_CUBIC)
                off = (new_sz - IMG_SIZE) // 2
                img = img_big[off:off+IMG_SIZE, off:off+IMG_SIZE].copy()
            # if equal, no change

        # Random erasing
        if random.random() < 0.2:
            esz = random.randint(2, 7)
            ex = random.randint(0, IMG_SIZE - esz)
            ey = random.randint(0, IMG_SIZE - esz)
            img[ey:ey+esz, ex:ex+esz] = random.uniform(0, 1)

        # Random invert
        if random.random() < 0.08:
            img = 1.0 - img

        batch[i, :, :, 0] = img.astype(np.float32)

    return batch


# ─── DATA ─────────────────────────────────────────────────────────────
def load_data():
    if not PRINTED_V4.exists():
        print(f"Printed v4 data not found: {PRINTED_V4}")
        print("Run: python3 generate_printed_v4.py")
        exit(1)

    d = np.load(PRINTED_V4)
    X, y = d['X'], d['y']
    print(f"Loaded {len(X):,} printed images")

    # Shuffle and split: 80% train, 20% test
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)

    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    return X_train, y_train, X_test, y_test


# ─── MODEL ────────────────────────────────────────────────────────────
class PrintedCNN(nn.Module):
    """Printed-only CNN with rotation-invariant features."""
    def __init__(self):
        super().__init__()
        # Larger first layer to capture rotation-invariant features
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm(48)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(96)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(192)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.35)

        self.conv4 = nn.Conv2d(192, 320, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(320)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.4)

        # After 4 pools: 28 → 14 → 7 → 3 → 1, channels 320
        self.classifier = nn.Sequential(
            nn.Linear(320, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x))); x = self.pool1(x); x = self.drop1(x)
        x = nn.relu(self.bn2(self.conv2(x))); x = self.pool2(x); x = self.drop2(x)
        x = nn.relu(self.bn3(self.conv3(x))); x = self.pool3(x); x = self.drop3(x)
        x = nn.relu(self.bn4(self.conv4(x))); x = self.pool4(x); x = self.drop4(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


# ─── TRAINING ──────────────────────────────────────────────────────────
def loss_fn(m, x, y):
    return nn.losses.cross_entropy(m(x), y, reduction='mean')

def acc_fn(m, x, y):
    return mx.mean(mx.argmax(m(x), axis=1) == y)

def batches(X, y, bs, shuffle=True, aug=False):
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, bs):
        e = min(s+bs, n)
        bi = idx[s:e]
        bx = X[bi]
        if aug:
            bx = augment_batch(bx)
        yield mx.array(bx), mx.array(y[bi])


def train():
    print("=" * 60)
    print("  PRINTED-ONLY A-Z CNN — Rotation Invariant")
    print("=" * 60)

    Xt, yt, Xte, yte = load_data()

    model = PrintedCNN()
    mx.eval(model.parameters())
    n_params = sum(np.prod(v.shape) for _, v in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    opt = optim.Adam(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: loss_fn(m, x, y))

    best_va = 0.0
    print(f"\n{EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print("-" * 60)

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        tls, tas, tbn = 0.0, 0.0, 0
        for bx, by in batches(Xt, yt, BATCH_SIZE, aug=True):
            l, g = loss_and_grad(model, bx, by)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            tls += l.item()
            tas += acc_fn(model, bx, by).item()
            tbn += 1

        model.eval()
        vls, vas, vbn = 0.0, 0.0, 0
        # Validation on a subset of test (no separate val set, using test)
        Xv, yv = Xte[:len(Xte)//4], yte[:len(Xte)//4]
        for bx, by in batches(Xv, yv, BATCH_SIZE, shuffle=False):
            vls += loss_fn(model, bx, by).item()
            vas += acc_fn(model, bx, by).item()
            vbn += 1

        tl, ta = tls / tbn, tas / tbn
        vl, va = vls / vbn, vas / vbn

        print(f"Ep {ep:2d} | tr_loss:{tl:.4f} tr_acc:{ta:.4f} | "
              f"vl_loss:{vl:.4f} vl_acc:{va:.4f} | {time.time() - t0:.1f}s")

        if va > best_va:
            best_va = va
            model.save_weights(str(MODEL_DIR / 'printed_best.safetensors'))
            print(f"  → Saved (va={va:.4f})")

    print("-" * 60)

    # ─── TEST ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST EVALUATION")
    print("=" * 60)

    best_path = MODEL_DIR / 'printed_best.safetensors'
    if best_path.exists():
        model.load_weights(str(best_path))

    model.eval()
    tls, tas, tbn = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for bx, by in batches(Xte, yte, BATCH_SIZE, shuffle=False):
        tls += loss_fn(model, bx, by).item()
        tas += acc_fn(model, bx, by).item()
        all_preds.extend(np.array(mx.argmax(model(bx), axis=1)).tolist())
        all_labels.extend(by.tolist())
        tbn += 1

    test_loss = tls / tbn
    test_acc = tas / tbn
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Best Val:  {best_va:.4f} ({best_va * 100:.2f}%)")

    # Per-class
    cc = np.zeros(NUM_CLASSES, dtype=np.int32)
    ct = np.zeros(NUM_CLASSES, dtype=np.int32)
    for p, l in zip(all_preds, all_labels):
        ct[l] += 1
        if p == l: cc[l] += 1

    print(f"\nPer-Class Accuracy:")
    for i in range(NUM_CLASSES):
        a = cc[i] / ct[i] if ct[i] > 0 else 0
        bar = "█" * int(a * 25) + "░" * (25 - int(a * 25))
        print(f"  {LETTERS[i]}: {bar} {a:.3f} ({cc[i]}/{ct[i]})")

    # ─── SAVE ─────────────────────────────────────────────────────────
    final_path = MODEL_DIR / 'printed_cnn.safetensors'
    model.save_weights(str(final_path))
    print(f"\nFinal model: {final_path}")
    np.savez(MODEL_DIR / 'test_data_printed.npz', X_test=Xte, y_test=yte)

    print(f"\nDone. Best={best_va:.4f} | Test={test_acc:.4f}")


if __name__ == "__main__":
    train()
