#!/usr/bin/env python3
"""
Test the trained A-Z Alphabet CNN model.
Loads saved model, runs predictions on test set, creates visual examples.

MLX 0.31.x: NHWC format, __call__ override, MaxPool2d modules.
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─── CONFIG ───────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A') + i) for i in range(NUM_CLASSES)]


# ─── MODEL DEFINITION (must match training) ────────────────────────────
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
            nn.Linear(256 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = nn.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


# ─── LOADING ───────────────────────────────────────────────────────────
def load_model():
    model = AlphabetCNN()
    mx.eval(model.parameters())
    model_path = MODEL_DIR / "alphabet_cnn_mlx.safetensors"

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Run train_alphabet_cnn.py first.")
        sys.exit(1)

    model.load_weights(str(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def load_test_data():
    test_path = MODEL_DIR / "test_data.npz"
    if not test_path.exists():
        print(f"Warning: test_data.npz not found at {test_path}")
        return None, None
    data = np.load(test_path)
    return data["X_test"], data["y_test"]


# ─── PREDICTION ────────────────────────────────────────────────────────
def predict(model, images):
    """Predict classes for input images.
    Args:
        images: numpy array shape (N, 28, 28, 1) or (N, 28, 28) or (28, 28)
    Returns:
        predicted_classes, probabilities
    """
    if images.ndim == 2:  # (28, 28)
        images = images[np.newaxis, :, :, np.newaxis]
    elif images.ndim == 3:
        if images.shape[-1] == 1:  # (N, 28, 28, 1)
            pass
        else:  # (N, 28, 28)
            images = images[..., np.newaxis]

    x = mx.array(images.astype(np.float32))
    logits = model(x)
    probs = mx.softmax(logits, axis=1)
    preds = mx.argmax(logits, axis=1)
    mx.eval(probs, preds)

    return np.array(preds), np.array(probs)


# ─── VISUALIZATION ─────────────────────────────────────────────────────
def create_sample_grid(model, X_test, y_test, samples_per_class=3):
    if not HAS_PIL:
        print("PIL not available, skipping visualization.")
        return

    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, label in enumerate(y_test):
        class_indices[label].append(idx)

    selected = []
    for label in range(NUM_CLASSES):
        if class_indices[label]:
            idxs = random.sample(class_indices[label],
                                 min(samples_per_class, len(class_indices[label])))
            selected.extend([(idx, label) for idx in idxs])

    random.shuffle(selected)

    batch_imgs = X_test[[idx for idx, _ in selected]]
    preds, probs = predict(model, batch_imgs)

    img_size = 56
    cols = 13
    rows = (len(selected) + cols - 1) // cols
    spacing = 4
    text_height = 30
    grid_w = cols * (img_size + spacing) + spacing
    grid_h = rows * (img_size + text_height + spacing) + spacing

    grid = Image.new("L", (grid_w, grid_h), color=255)
    draw = ImageDraw.Draw(grid)

    for i, ((idx, true_label), pred, prob) in enumerate(zip(selected, preds, probs)):
        row = i // cols
        col = i % cols
        x = spacing + col * (img_size + spacing)
        y = spacing + row * (img_size + text_height + spacing)

        # X_test is NHWC: (N, 28, 28, 1)
        img = X_test[idx][:, :, 0] if X_test[idx].ndim == 3 else X_test[idx]
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode="L")
        img_pil = img_pil.resize((img_size, img_size), Image.NEAREST)
        grid.paste(img_pil, (x, y))

        true_letter = LETTERS[true_label]
        pred_letter = LETTERS[pred]
        max_prob = np.max(prob) * 100
        color = "green" if true_label == pred else "red"
        text = f"{true_letter}->{pred_letter} ({max_prob:.0f}%)"
        draw.text((x, y + img_size + 2), text, fill=0)

    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    grid_path = output_dir / "predictions_grid.png"
    grid.save(grid_path)
    print(f"Sample predictions grid saved to: {grid_path}")


def create_confusion_chart(model, X_test, y_test):
    preds, probs = predict(model, X_test)

    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX (per-class accuracy)")
    print("=" * 60)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for p, t in zip(preds, y_test):
        confusion[t, p] += 1

    for i in range(NUM_CLASSES):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / total if total > 0 else 0
        top_errors = []
        for j in range(NUM_CLASSES):
            if i != j and confusion[i, j] > 0:
                top_errors.append((LETTERS[j], confusion[i, j]))
        top_errors.sort(key=lambda x: -x[1])
        error_str = ", ".join(f"{l}:{c}" for l, c in top_errors[:3])
        bar = "█" * int(acc * 30) + "░" * (30 - int(acc * 30))
        print(f"  {LETTERS[i]}: {bar} {acc:.3f}  (errors: {error_str if error_str else 'none'})")

    return confusion


def test_individual_samples(model, X_test, y_test, num_samples=15):
    indices = random.sample(range(len(X_test)), min(num_samples, len(X_test)))

    print("\n" + "=" * 60)
    print(f"  INDIVIDUAL SAMPLE PREDICTIONS ({num_samples} random)")
    print("=" * 60)
    print(f"  {'#':<5} {'True':<6} {'Pred':<6} {'Confidence':<12} {'Top-3 Predictions'}")
    print("-" * 60)

    batch_imgs = X_test[indices]
    preds, probs = predict(model, batch_imgs)

    for i, (idx, pred, prob) in enumerate(zip(indices, preds, probs)):
        true_label = y_test[idx]
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_str = "  ".join(f"{LETTERS[j]}:{prob[j]*100:.1f}%" for j in top3_idx)

        status = "✓" if true_label == pred else "✗"
        print(f"  {status} {i+1:<4} {LETTERS[true_label]:<6} {LETTERS[pred]:<6} "
              f"{prob[pred]*100:.1f}%{'':<7} {top3_str}")

    correct = sum(1 for idx, pred in zip(indices, preds) if y_test[idx] == pred)
    print("-" * 60)
    print(f"  Accuracy on {num_samples} samples: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")


def create_letter_grid():
    if not HAS_PIL:
        return

    model = load_model()
    X_test, y_test = load_test_data()
    if X_test is None:
        print("No test data available.")
        return

    class_indices = {i: [] for i in range(NUM_CLASSES)}
    for idx, label in enumerate(y_test):
        class_indices[label].append(idx)

    selected = []
    for label in range(NUM_CLASSES):
        indices = random.sample(class_indices[label],
                                min(20, len(class_indices[label])))
        batch = X_test[indices]
        preds, probs = predict(model, batch)

        best_idx = None
        best_confidence = -1
        for j, (pred, prob) in enumerate(zip(preds, probs)):
            if pred == label and prob[pred] > best_confidence:
                best_confidence = prob[pred]
                best_idx = indices[j]

        if best_idx is not None:
            selected.append((best_idx, label))

    img_size = 56
    cols = 13
    rows = 2
    spacing = 8
    grid_w = cols * (img_size + spacing) + spacing
    grid_h = rows * (img_size + spacing + 25) + spacing

    grid = Image.new("L", (grid_w, grid_h), color=255)
    draw = ImageDraw.Draw(grid)

    for i, (idx, label) in enumerate(selected[:26]):
        row = i // cols
        col = i % cols
        x = spacing + col * (img_size + spacing)
        y = spacing + row * (img_size + spacing + 25)

        img = X_test[idx][:, :, 0] if X_test[idx].ndim == 3 else X_test[idx]
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode="L")
        img_pil = img_pil.resize((img_size, img_size), Image.NEAREST)
        grid.paste(img_pil, (x, y))
        draw.text((x + img_size // 2 - 4, y + img_size + 2), LETTERS[label], fill=0)

    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    grid_path = output_dir / "alphabet_grid.png"
    grid.save(grid_path)
    print(f"Alphabet grid saved to: {grid_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  A-Z HANDWRITTEN ALPHABET CNN - TEST & EVALUATION")
    print("=" * 60)

    model = load_model()
    X_test, y_test = load_test_data()

    if X_test is not None:
        print(f"Test data: {X_test.shape[0]} samples")

        print("\nRunning full test evaluation...")
        preds, probs = predict(model, X_test)
        accuracy = np.mean(preds == y_test)
        print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        confusion = create_confusion_chart(model, X_test, y_test)
        test_individual_samples(model, X_test, y_test, num_samples=20)

        if HAS_PIL:
            print("\nCreating visualizations...")
            create_sample_grid(model, X_test, y_test, samples_per_class=2)
            create_letter_grid()

    print("\n" + "=" * 60)
    print("  INTERACTIVE DEMO")
    print("=" * 60)
    print("  Enter a letter (A-Z) to see random test predictions for that letter.")
    print("  Type 'random' for random samples, 'grid' to rebuild grid, or 'quit' to exit.")

    while X_test is not None:
        try:
            cmd = input("\n> ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("QUIT", "EXIT", "Q"):
            break
        elif cmd == "GRID":
            create_letter_grid()
        elif cmd == "RANDOM":
            test_individual_samples(model, X_test, y_test, num_samples=10)
        elif cmd in LETTERS:
            label = LETTERS.index(cmd)
            indices = [i for i, l in enumerate(y_test) if l == label]
            if indices:
                sample_indices = random.sample(indices, min(5, len(indices)))
                batch = X_test[sample_indices]
                preds, probs = predict(model, batch)
                print(f"\n  Predictions for letter '{cmd}':")
                for idx, pred, prob in zip(sample_indices, preds, probs):
                    status = "✓" if pred == label else "✗"
                    top3 = np.argsort(prob)[::-1][:3]
                    top3_str = ", ".join(f"{LETTERS[j]}:{prob[j]*100:.0f}%" for j in top3)
                    print(f"    {status} Predicted: {LETTERS[pred]} | Top-3: {top3_str}")
            else:
                print(f"  No test samples for '{cmd}'")
        else:
            print(f"  Unknown command. Use a letter (A-Z), 'random', 'grid', or 'quit'.")


if __name__ == "__main__":
    main()
