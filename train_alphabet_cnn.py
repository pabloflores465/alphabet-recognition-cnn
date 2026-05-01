#!/usr/bin/env python3
"""
A-Z Handwritten Alphabet Classification - CNN with MLX (Apple Silicon)
Dataset: Kaggle A-Z Handwritten Alphabets in .csv format
~372K images, 28x28 grayscale, 26 classes (A-Z)
Optimized for Apple M1 with 8GB RAM

MLX 0.31.x specifics:
- Image format: NHWC (batch, height, width, channels)
- Conv2d weight: (out_ch, kH, kW, in_ch)
- Use nn.MaxPool2d (module), not nn.max_pool2d (function)
- Override __call__ not forward
"""

import os
import sys
import csv
import time
import pickle
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
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.05
NUM_CLASSES = 26
IMG_SIZE = 28
RANDOM_SEED = 42


# ─── DATA LOADING ─────────────────────────────────────────────────────
def load_and_preprocess():
    """Load CSV, preprocess into NHWC format for MLX."""
    print(f"Loading dataset from {CSV_FILE}...")
    print(f"File size: {CSV_FILE.stat().st_size / 1024**2:.0f} MB")

    data = []
    labels_list = []

    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if len(row) < 785:
                continue
            try:
                label = int(row[0])
                pixels = np.array(row[1:785], dtype=np.float32) / 255.0
                data.append(pixels)
                labels_list.append(label)
            except (ValueError, IndexError):
                continue

            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i+1} samples...")

    X = np.array(data, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    del data, labels_list

    # MLX uses NHWC: (batch, height, width, channels)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(f"Loaded {X.shape[0]} samples, shape={X.shape}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Class distribution: min={np.min(np.bincount(y))}, max={np.max(np.bincount(y))}")
    return X, y


def split_data(X, y):
    """Shuffle and split into train/val/test."""
    np.random.seed(RANDOM_SEED)
    n = len(X)
    indices = np.random.permutation(n)

    test_size = int(n * TEST_SPLIT)
    val_size = int(n * VALIDATION_SPLIT)
    train_size = n - test_size - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


# ─── CNN MODEL ─────────────────────────────────────────────────────────
class AlphabetCNN(nn.Module):
    """CNN for 28x28 grayscale alphabet classification (26 classes).
    Input: NHWC (batch, 28, 28, 1)"""

    def __init__(self):
        super().__init__()

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head
        # After 4 max-pools (28 -> 14 -> 7 -> 3 -> 1), 256 channels
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def __call__(self, x):
        # Block 1: 28 -> 14
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.pool1(x)

        # Block 2: 14 -> 7
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.pool2(x)

        # Block 3: 7 -> 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.relu(x)
        x = self.pool3(x)

        # Block 4: 3 -> 1
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.relu(x)
        x = self.pool4(x)

        # Flatten and classify
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


# ─── TRAINING UTILITIES ──────────────────────────────────────────────
def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction='mean')

def accuracy_fn(model, x, y):
    logits = model(x)
    preds = mx.argmax(logits, axis=1)
    return mx.mean(preds == y)


def iterate_batches(X, y, batch_size, shuffle=True):
    n = len(X)
    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield mx.array(X[batch_idx]), mx.array(y[batch_idx])


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  A-Z Handwritten Alphabet CNN Classifier (MLX on Apple Silicon)")
    print("=" * 60)

    # Load and preprocess
    X, y = load_and_preprocess()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create model
    model = AlphabetCNN()
    mx.eval(model.parameters())

    # Count parameters
    total_params = sum(
        np.prod(v.shape) for _, v in tree_flatten(model.parameters())
    )
    print(f"\nModel parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    # value_and_grad wrapper
    # nn.value_and_grad(model, fn) -> inner_fn(model, *args) -> fn(model, *args)
    # So compute_loss receives (model, x, y) directly
    def compute_loss(m, x, y):
        return loss_fn(m, x, y)

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_params = None

    print(f"\nStarting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Training
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for batch_x, batch_y in iterate_batches(X_train, y_train, BATCH_SIZE):
            loss_val, grads = loss_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            acc_val = accuracy_fn(model, batch_x, batch_y)
            train_loss_sum += loss_val.item()
            train_acc_sum += acc_val.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches
        train_acc = train_acc_sum / train_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

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
              f"time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint of best model
            # Save checkpoint of best model
            model.save_weights(str(MODEL_DIR / 'alphabet_cnn_best.safetensors'))

    print("-" * 60)

    # ─── TEST EVALUATION ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    # Load best checkpoint if available
    best_path = MODEL_DIR / 'alphabet_cnn_best.safetensors'
    if best_path.exists():
        model.load_weights(str(best_path))
        print("  Loaded best checkpoint for evaluation")

    model.eval()
    test_loss_sum = 0.0
    test_acc_sum = 0.0
    test_batches = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in iterate_batches(X_test, y_test, BATCH_SIZE, shuffle=False):
        loss_val = loss_fn(model, batch_x, batch_y)
        acc_val = accuracy_fn(model, batch_x, batch_y)
        logits = model(batch_x)
        preds = mx.argmax(logits, axis=1)

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

    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int32)
    class_total = np.zeros(NUM_CLASSES, dtype=np.int32)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    letters = [chr(ord('A') + i) for i in range(NUM_CLASSES)]
    for i in range(NUM_CLASSES):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {letters[i]}: {bar} {acc:.3f} ({class_correct[i]}/{class_total[i]})")

    # ─── SAVE MODEL ─────────────────────────────────────────────────
    model_path = MODEL_DIR / "alphabet_cnn_mlx.safetensors"
    model.save_weights(str(model_path))
    print(f"\n  Model saved to: {model_path}")

    history_path = MODEL_DIR / "training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    np.savez(MODEL_DIR / "test_data.npz", X_test=X_test, y_test=y_test)
    print(f"  History saved to: {history_path}")
    print(f"  Test data saved to: {MODEL_DIR / 'test_data.npz'}")

    # ─── FINAL SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test accuracy:           {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Model:                   {total_params:,} parameters")
    print(f"  Dataset:                 372,450 images (A-Z)")
    print(f"  Device:                  Apple Silicon (MLX + Metal)")
    print("=" * 60)

    return model, history, best_val_acc


if __name__ == "__main__":
    train()
