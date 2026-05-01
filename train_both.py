#!/usr/bin/env python3
"""
Definitive A-Z CNN — Handwritten + Printed (balanced)
Same architecture as alphabet_cnn_mlx.safetensors so it's a drop-in replacement.
Targets both "letra de carta" (handwritten) and "letra de molde" (printed).
"""

import csv, time, random, numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ─── CONFIG ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = Path.home() / ".cache/kagglehub/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/versions/5"
CSV_FILE = DATA_DIR / "A_Z Handwritten Data.csv"
PRINTED_V2 = PROJECT_DIR / "printed_dataset_v2" / "printed_letters_v2.npz"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]

# ─── AUGMENTATION ─────────────────────────────────────────────────────
def augment_batch(images_np):
    batch = images_np.copy()
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.6, 1.5)
        batch = np.clip((batch - 0.5) * contrast + 0.5 * brightness, 0, 1)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(0.005, 0.04), batch.shape).astype(np.float32)
        batch = np.clip(batch + noise, 0, 1)
    for i in range(batch.shape[0]):
        img = batch[i, :, :, 0]
        if random.random() < 0.5:
            img = np.roll(img, random.randint(-2, 2), axis=1)
            img = np.roll(img, random.randint(-2, 2), axis=0)
        if random.random() < 0.1:
            img = 1.0 - img
        batch[i, :, :, 0] = img
    return batch


# ─── DATA ─────────────────────────────────────────────────────────────
def load_handwritten():
    print("Loading handwritten dataset...")
    data, labels = [], []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) < 785: continue
            try:
                labels.append(int(row[0]))
                data.append(np.array(row[1:785], dtype=np.float32) / 255.0)
            except: continue
            if (i+1) % 50000 == 0: print(f"  {i+1}...")
    X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype=np.int32)
    print(f"  Handwritten: {len(X):,} samples")
    return X, y


def load_printed():
    if not PRINTED_V2.exists():
        print(f"Printed data not found at {PRINTED_V2}")
        print("Run: python3 generate_printed_data_v2.py")
        return np.array([]), np.array([])
    d = np.load(PRINTED_V2)
    X, y = d['X'], d['y']
    print(f"  Printed v2: {len(X):,} samples")
    return X, y


def load_balanced():
    """Load handwritten + printed, oversample printed to balance."""
    Xh, yh = load_handwritten()
    Xp, yp = load_printed()

    # Oversample printed to ~30% of handwritten (so model learns both)
    if len(Xp) > 0:
        target_printed = len(Xh) // 3  # ~124K printed
        repeats = max(1, target_printed // len(Xp))
        Xp = np.tile(Xp, (repeats, 1, 1, 1))
        yp = np.tile(yp, repeats)
        # Trim to exact target
        Xp, yp = Xp[:target_printed], yp[:target_printed]

    X = np.concatenate([Xh, Xp])
    y = np.concatenate([yh, yp])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"  Total: {len(X):,} samples (handwritten + oversampled printed)")
    print(f"  Class balance: {np.bincount(y).min()} - {np.bincount(y).max()}")
    return X, y


def split(X, y, val_pct=0.1, test_pct=0.05):
    np.random.seed(42)
    n = len(X)
    idx = np.random.permutation(n)
    ts = int(n * test_pct)
    vs = int(n * val_pct)
    tr = n - ts - vs
    return (X[idx[:tr]], y[idx[:tr]],
            X[idx[tr:tr+vs]], y[idx[tr:tr+vs]],
            X[idx[tr+vs:]], y[idx[tr+vs:]])


# ─── MODEL (same architecture as original 99.2% model) ────────────────
class AlphabetCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout * 0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(dropout * 0.5)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(dropout * 0.5)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(dropout * 0.5)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
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
        if aug: bx = augment_batch(bx)
        yield mx.array(bx), mx.array(y[bi])


def train():
    print("="*60)
    print("  A-Z CNN — HANDWRITTEN + PRINTED (DEFINITIVE)")
    print("="*60)

    X, y = load_balanced()
    Xt, yt, Xv, yv, Xte, yte = split(X, y)
    print(f"Train: {len(Xt):,}  Val: {len(Xv):,}  Test: {len(Xte):,}")

    model = AlphabetCNN(dropout=0.35)
    mx.eval(model.parameters())

    # Optionally start from pretrained handwritten weights
    pretrained_path = MODEL_DIR / "alphabet_cnn_mlx.safetensors"
    if pretrained_path.exists():
        print("Loading pretrained handwritten weights as starting point...")
        model.load_weights(str(pretrained_path))

    n_params = sum(np.prod(v.shape) for _, v in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    opt = optim.Adam(learning_rate=LEARNING_RATE)

    def loss_wrapper(m, x, y):
        return loss_fn(m, x, y)

    loss_and_grad = nn.value_and_grad(model, loss_wrapper)

    best_va = 0.0
    print(f"\n{EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("-"*60)

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        tls, tas, tbn = 0.0, 0.0, 0
        for bx, by in batches(Xt, yt, BATCH_SIZE, aug=True):
            l, g = loss_and_grad(model, bx, by)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            tls += l.item(); tas += acc_fn(model, bx, by).item(); tbn += 1

        model.eval()
        vls, vas, vbn = 0.0, 0.0, 0
        for bx, by in batches(Xv, yv, BATCH_SIZE, shuffle=False):
            vls += loss_fn(model, bx, by).item()
            vas += acc_fn(model, bx, by).item(); vbn += 1

        tl, ta = tls/tbn, tas/tbn
        vl, va = vls/vbn, vas/vbn

        print(f"Ep {ep:2d} | tr_loss:{tl:.4f} tr_acc:{ta:.4f} | "
              f"vl_loss:{vl:.4f} vl_acc:{va:.4f} | {time.time()-t0:.1f}s")

        if va > best_va:
            best_va = va
            model.save_weights(str(MODEL_DIR / 'alphabet_both_best.safetensors'))
            print(f"  → Saved best (va={va:.4f})")

    print("-"*60)

    # ─── TEST ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEST EVALUATION")
    print("="*60)

    best_path = MODEL_DIR / 'alphabet_both_best.safetensors'
    if best_path.exists():
        model.load_weights(str(best_path))
        print("Loaded best checkpoint")

    model.eval()
    tls, tas, tbn = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for bx, by in batches(Xte, yte, BATCH_SIZE, shuffle=False):
        tls += loss_fn(model, bx, by).item()
        tas += acc_fn(model, bx, by).item()
        all_preds.extend(np.array(mx.argmax(model(bx), axis=1)).tolist())
        all_labels.extend(by.tolist())
        tbn += 1

    test_loss = tls/tbn
    test_acc = tas/tbn
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Best Val:  {best_va:.4f} ({best_va*100:.2f}%)")

    # Per-class
    cc = np.zeros(NUM_CLASSES, dtype=np.int32)
    ct = np.zeros(NUM_CLASSES, dtype=np.int32)
    for p, l in zip(all_preds, all_labels):
        ct[l] += 1
        if p == l: cc[l] += 1

    print(f"\nPer-Class Accuracy:")
    for i in range(NUM_CLASSES):
        a = cc[i]/ct[i] if ct[i]>0 else 0
        bar = "█"*int(a*25) + "░"*(25-int(a*25))
        print(f"  {LETTERS[i]}: {bar} {a:.3f} ({cc[i]}/{ct[i]})")

    # ─── SAVE FINAL ───────────────────────────────────────────────────
    final_path = MODEL_DIR / "alphabet_both.safetensors"
    model.save_weights(str(final_path))
    print(f"\nFinal model: {final_path}")

    # Also save test data
    np.savez(MODEL_DIR / "test_data_both.npz", X_test=Xte, y_test=yte)

    print(f"\nDone. Best Val={best_va:.4f} | Test={test_acc:.4f}")
    return model


if __name__ == "__main__":
    train()
