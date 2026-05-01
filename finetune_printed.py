#!/usr/bin/env python3
"""
Quick fine-tune: take pretrained handwritten model, fine-tune on printed v3.
Target: add printed character recognition without losing handwritten accuracy.
"""

import numpy as np
import time, random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
PRINTED_V3 = PROJECT_DIR / "printed_dataset_v3" / "printed_letters_v3.npz"
NUM_CLASSES = 26
IMG_SIZE = 28
LETTERS = [chr(ord('A')+i) for i in range(NUM_CLASSES)]
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.0003


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


def augment(batch):
    b = batch.copy()
    if random.random() < 0.5:
        b = np.clip((b - 0.5) * random.uniform(0.7, 1.3) + 0.5 * random.uniform(0.8, 1.2), 0, 1)
    if random.random() < 0.4:
        b = np.clip(b + np.random.normal(0, random.uniform(0.005, 0.05), b.shape).astype(np.float32), 0, 1)
    for i in range(b.shape[0]):
        img = b[i, :, :, 0]
        if random.random() < 0.5:
            img = np.roll(np.roll(img, random.randint(-2, 2), axis=1), random.randint(-2, 2), axis=0)
        b[i, :, :, 0] = img
    return b


def main():
    print("="*60)
    print("  FINE-TUNE: Handwritten → Printed v3 (grayscale)")
    print("="*60)

    # Load printed v3
    d = np.load(PRINTED_V3)
    X, y = d['X'], d['y']
    print(f"Printed v3: {len(X)} samples")

    # Split: 85% train, 15% val
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(n * 0.85)
    Xt, yt = X[idx[:n_train]], y[idx[:n_train]]
    Xv, yv = X[idx[n_train:]], y[idx[n_train:]]
    print(f"Train: {len(Xt)}, Val: {len(Xv)}")

    # Load pretrained handwritten model
    model = AlphabetCNN()
    mx.eval(model.parameters())
    pretrained = MODEL_DIR / "alphabet_cnn_mlx.safetensors"
    model.load_weights(str(pretrained))
    print(f"Loaded pretrained: {pretrained.name}")

    # Freeze early layers, train only classifier + last conv block
    # (Simpler: just train all with low LR)
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(m, x, y):
        return nn.losses.cross_entropy(m(x), y, reduction='mean')

    def acc_fn(m, x, y):
        return mx.mean(mx.argmax(m(x), axis=1) == y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    best_va = 0.0
    print(f"\n{EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
    print("-"*60)

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()

        # Shuffle train
        idx = np.random.permutation(len(Xt))
        tls, tas, n_batch = 0.0, 0.0, 0
        for s in range(0, len(Xt), BATCH_SIZE):
            e = min(s+BATCH_SIZE, len(Xt))
            bi = idx[s:e]
            bx = augment(Xt[bi])
            by = yt[bi]
            l, g = loss_and_grad(model, mx.array(bx), mx.array(by))
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)
            tls += l.item()
            tas += acc_fn(model, mx.array(bx), mx.array(by)).item()
            n_batch += 1

        model.eval()
        vls, vas, v_batch = 0.0, 0.0, 0
        for s in range(0, len(Xv), BATCH_SIZE):
            e = min(s+BATCH_SIZE, len(Xv))
            bx, by = mx.array(Xv[s:e]), mx.array(yv[s:e])
            vls += loss_fn(model, bx, by).item()
            vas += acc_fn(model, bx, by).item()
            v_batch += 1

        tl, ta = tls/n_batch, tas/n_batch
        vl, va = vls/v_batch, vas/v_batch
        print(f"Ep {ep:2d} | tr_loss:{tl:.4f} tr_acc:{ta:.4f} | "
              f"vl_loss:{vl:.4f} vl_acc:{va:.4f} | {time.time()-t0:.1f}s")

        if va > best_va:
            best_va = va
            model.save_weights(str(MODEL_DIR / 'alphabet_finetuned_best.safetensors'))

    # Save final
    model.save_weights(str(MODEL_DIR / 'alphabet_finetuned.safetensors'))
    print(f"\nBest val acc: {best_va:.4f}")
    print(f"Models saved: alphabet_finetuned.safetensors / alphabet_finetuned_best.safetensors")


if __name__ == "__main__":
    main()
