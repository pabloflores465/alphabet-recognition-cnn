#!/usr/bin/env python3
"""
Robust A-Z Alphabet CNN with combined dataset (handwritten + printed).
Uses heavy data augmentation for webcam-ready robustness.
Optimized for Apple M1 with MLX.
"""

import os
import sys
import csv
import time
import pickle
import random
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ─── CONFIG ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = Path.home() / ".cache/kagglehub/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/versions/5"
CSV_FILE = DATA_DIR / "A_Z Handwritten Data.csv"
PRINTED_DATA = PROJECT_DIR / "printed_dataset" / "printed_letters.npz"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.05
NUM_CLASSES = 26
IMG_SIZE = 28
RANDOM_SEED = 42
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]


# ─── DATA AUGMENTATION ─────────────────────────────────────────────
def augment_batch(images_np):
    """Apply fast random augmentations to a batch (numpy-only).
    Expects NHWC format: (batch, 28, 28, 1)"""
    batch = images_np.copy()
    B = batch.shape[0]
    
    # Batch-level ops (fast)
    if random.random() < 0.5:
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.6, 1.5)
        batch = np.clip((batch - 0.5) * contrast + 0.5 * brightness, 0, 1)
    
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(0.005, 0.04), batch.shape).astype(np.float32)
        batch = np.clip(batch + noise, 0, 1)
    
    # Per-image fast ops
    for i in range(B):
        img = batch[i, :, :, 0]
        # Random shift
        if random.random() < 0.5:
            img = np.roll(img, random.randint(-2, 2), axis=1)
            img = np.roll(img, random.randint(-2, 2), axis=0)
        # Random invert
        if random.random() < 0.1:
            img = 1.0 - img
        batch[i, :, :, 0] = img
    
    return batch


# ─── DATA LOADING ─────────────────────────────────────────────────────
def load_handwritten():
    """Load handwritten dataset from CSV."""
    print(f"Loading handwritten dataset...")
    data, labels = [], []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if len(row) < 785:
                continue
            try:
                labels.append(int(row[0]))
                data.append(np.array(row[1:785], dtype=np.float32) / 255.0)
            except (ValueError, IndexError):
                continue
            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i+1} handwritten...")
    X = np.array(data, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels, dtype=np.int32)
    print(f"  Handwritten: {X.shape[0]} samples")
    return X, y


def load_printed():
    """Load printed dataset."""
    if not PRINTED_DATA.exists():
        print("Printed dataset not found. Run generate_printed_data.py first.")
        return np.array([]), np.array([])
    data = np.load(PRINTED_DATA)
    X, y = data['X'], data['y']
    print(f"  Printed: {X.shape[0]} samples")
    return X, y


def load_combined():
    """Load and combine both datasets."""
    X_hw, y_hw = load_handwritten()
    X_pt, y_pt = load_printed()

    if len(X_pt) > 0:
        X = np.concatenate([X_hw, X_pt], axis=0)
        y = np.concatenate([y_hw, y_pt], axis=0)
    else:
        X, y = X_hw, y_hw

    print(f"Combined: {X.shape[0]} samples")
    return X, y


def split_data(X, y):
    """Shuffle and split into train/val/test."""
    np.random.seed(RANDOM_SEED)
    n = len(X)
    indices = np.random.permutation(n)
    test_sz = int(n * TEST_SPLIT)
    val_sz = int(n * VALIDATION_SPLIT)
    train_sz = n - test_sz - val_sz
    return (X[indices[:train_sz]], y[indices[:train_sz]],
            X[indices[train_sz:train_sz+val_sz]], y[indices[train_sz:train_sz+val_sz]],
            X[indices[train_sz+val_sz:]], y[indices[train_sz+val_sz:]])


# ─── CNN MODEL ─────────────────────────────────────────────────────────
class RobustAlphabetCNN(nn.Module):
    """Robust CNN for alphabet classification.
    Input: NHWC (batch, 28, 28, 1)"""

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        d = dropout_rate

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        self.drop1 = nn.Dropout(d * 0.4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14 -> 7
        self.drop2 = nn.Dropout(d * 0.6)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7 -> 3
        self.drop3 = nn.Dropout(d * 0.8)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3 -> 1
        self.drop4 = nn.Dropout(d)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(d * 0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = nn.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = nn.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


# ─── TRAINING UTILITIES ──────────────────────────────────────────────
def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction='mean')

def accuracy_fn(model, x, y):
    preds = mx.argmax(model(x), axis=1)
    return mx.mean(preds == y)


def iterate_batches(X, y, batch_size, shuffle=True, augment=False):
    n = len(X)
    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batch_x = X[batch_idx]
        if augment:
            batch_x = augment_batch(batch_x)
        yield mx.array(batch_x), mx.array(y[batch_idx])


# ─── MAIN ──────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  ROBUST A-Z CNN - Combined Handwritten + Printed")
    print("  MLX on Apple Silicon")
    print("=" * 60)

    X, y = load_combined()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = RobustAlphabetCNN(dropout_rate=0.4)
    mx.eval(model.parameters())

    total_params = sum(
        np.prod(v.shape) for _, v in tree_flatten(model.parameters())
    )
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    def compute_loss(m, x, y):
        return loss_fn(m, x, y)

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        train_loss_sum, train_acc_sum, train_batches = 0.0, 0.0, 0
        for batch_x, batch_y in iterate_batches(X_train, y_train, BATCH_SIZE, augment=True):
            loss_val, grads = loss_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            acc_val = accuracy_fn(model, batch_x, batch_y)
            train_loss_sum += loss_val.item()
            train_acc_sum += acc_val.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches
        train_acc = train_acc_sum / train_batches

        model.eval()
        val_loss_sum, val_acc_sum, val_batches = 0.0, 0.0, 0
        for batch_x, batch_y in iterate_batches(X_val, y_val, BATCH_SIZE, shuffle=False):
            loss_val = loss_fn(model, batch_x, batch_y)
            acc_val = accuracy_fn(model, batch_x, batch_y)
            val_loss_sum += loss_val.item()
            val_acc_sum += acc_val.item()
            val_batches += 1

        val_loss = val_loss_sum / val_batches
        val_acc = val_acc_sum / val_batches

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(str(MODEL_DIR / 'alphabet_robust_best.safetensors'))

    print("-" * 60)

    # ─── TEST ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST EVALUATION")
    print("=" * 60)

    best_path = MODEL_DIR / 'alphabet_robust_best.safetensors'
    if best_path.exists():
        model.load_weights(str(best_path))

    model.eval()
    test_loss_sum, test_acc_sum, test_batches = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for batch_x, batch_y in iterate_batches(X_test, y_test, BATCH_SIZE, shuffle=False):
        loss_val = loss_fn(model, batch_x, batch_y)
        acc_val = accuracy_fn(model, batch_x, batch_y)
        preds = mx.argmax(model(batch_x), axis=1)
        test_loss_sum += loss_val.item()
        test_acc_sum += acc_val.item()
        test_batches += 1
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.tolist())

    test_loss = test_loss_sum / test_batches
    test_acc = test_acc_sum / test_batches
    print(f"\n  Test Loss:  {test_loss:.4f}")
    print(f"  Test Acc:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Best Val:   {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # Per-class
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int32)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int32)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    print(f"\n  Per-Class Accuracy:")
    for i in range(NUM_CLASSES):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {LETTERS[i]}: {bar} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

    # ─── SAVE ──────────────────────────────────────────────────────
    final_path = MODEL_DIR / "alphabet_robust.safetensors"
    model.save_weights(str(final_path))
    print(f"\n  Final model: {final_path}")

    history_path = MODEL_DIR / "training_history_robust.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    np.savez(MODEL_DIR / "test_data_robust.npz", X_test=X_test, y_test=y_test)
    print(f"  History: {history_path}")

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val: {best_val_acc:.4f} | Test: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
