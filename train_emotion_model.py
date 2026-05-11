"""
EVA Emotion Model Trainer (PyTorch + MobileNetV2)
- Handles class imbalance with weighted sampling
- Transfer learning from ImageNet
- Heavy augmentation for FER-2013
- Exports to ONNX for use in emotion_detector.py
"""
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────────
TRAIN_DIR    = "images/train"
VAL_DIR      = "images/validation"
IMG_SIZE     = 96
BATCH_SIZE   = 64
EPOCHS_HEAD  = 8
EPOCHS_FINE  = 20
LR_HEAD      = 1e-3
LR_FINE      = 1e-5
NUM_WORKERS  = 0      # Windows: keep 0 to avoid spawn issues
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR      = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODEL_NAME   = "eva_emotion_v1"

print(f"\n{'='*60}")
print(f"🎓 EVA Emotion Model Training (PyTorch)")
print(f"{'='*60}")
print(f"Device:     {DEVICE}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")

# ─── Data Augmentation ──────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std =[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

CLASSES = train_ds.classes
NUM_CLASSES = len(CLASSES)
print(f"📊 Classes: {CLASSES}")
print(f"   Train:  {len(train_ds)} images")
print(f"   Val:    {len(val_ds)} images")

# ─── Class-balanced sampler (fixes disgust imbalance) ───────────────────
class_counts = Counter(train_ds.targets)
print(f"\n📈 Class distribution & sampling weights:")
class_weights = []
for i, c in enumerate(CLASSES):
    w = 1.0 / class_counts[i]
    class_weights.append(w)
    print(f"   {c:<10} count={class_counts[i]:<6} weight={w:.6f}")

# Per-sample weights for the sampler
sample_weights = [class_weights[t] for t in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=False)

# ─── Model: MobileNetV2 transfer learning ───────────────────────────────
print(f"\n🏗️  Loading MobileNetV2 (ImageNet weights)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Replace classifier head for 7 emotions
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES),
)

# Phase 1: freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(DEVICE)
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params:     {total_params:,}")
print(f"   Trainable (head): {trainable_params:,}")

# ─── Loss & Optimizer ───────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_HEAD)

# ─── Training & Eval Loops ──────────────────────────────────────────────
def train_one_epoch(loader, desc=""):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total   += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{100*correct/total:.1f}%")
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(loader, collect_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total   += imgs.size(0)
        if collect_preds:
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if collect_preds:
        return total_loss / total, correct / total, all_preds, all_labels
    return total_loss / total, correct / total


# ─── PHASE 1: Train head only ───────────────────────────────────────────
print(f"\n{'='*60}")
print(f"🚀 PHASE 1: Training classifier head ({EPOCHS_HEAD} epochs)")
print(f"{'='*60}")

best_val_acc = 0.0
for epoch in range(1, EPOCHS_HEAD + 1):
    train_loss, train_acc = train_one_epoch(train_loader,
                                            desc=f"Epoch {epoch}/{EPOCHS_HEAD}")
    val_loss, val_acc = evaluate(val_loader)
    print(f"  Epoch {epoch:2d}/{EPOCHS_HEAD} | "
          f"train_loss={train_loss:.3f} acc={train_acc*100:.1f}% | "
          f"val_loss={val_loss:.3f} acc={val_acc*100:.1f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT_DIR / f"{MODEL_NAME}_best.pt")
        print(f"     💾 Saved best (val_acc={val_acc*100:.2f}%)")

# ─── PHASE 2: Fine-tune deeper layers ───────────────────────────────────
print(f"\n{'='*60}")
print(f"🔬 PHASE 2: Fine-tuning ({EPOCHS_FINE} epochs)")
print(f"{'='*60}")

# Unfreeze last few feature blocks
for param in model.features[-4:].parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Trainable (after unfreeze): {trainable_params:,}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_FINE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                  factor=0.5, patience=3)

patience_counter = 0
PATIENCE = 6
for epoch in range(1, EPOCHS_FINE + 1):
    train_loss, train_acc = train_one_epoch(train_loader,
                                            desc=f"Fine {epoch}/{EPOCHS_FINE}")
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_acc)
    print(f"  Epoch {epoch:2d}/{EPOCHS_FINE} | "
          f"train_loss={train_loss:.3f} acc={train_acc*100:.1f}% | "
          f"val_loss={val_loss:.3f} acc={val_acc*100:.1f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT_DIR / f"{MODEL_NAME}_best.pt")
        print(f"     💾 Saved best (val_acc={val_acc*100:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"     ⏹️  Early stopping (no improvement for {PATIENCE} epochs)")
            break

# ─── Load best & final eval ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"📊 FINAL EVALUATION (best model, val_acc={best_val_acc*100:.2f}%)")
print(f"{'='*60}")
model.load_state_dict(torch.load(OUT_DIR / f"{MODEL_NAME}_best.pt",
                                 map_location=DEVICE))

final_loss, final_acc, preds, labels = evaluate(val_loader, collect_preds=True)
print(f"\n  Validation Loss:     {final_loss:.4f}")
print(f"  Validation Accuracy: {final_acc*100:.2f}%\n")

print("📈 Per-class report:")
print(classification_report(labels, preds, target_names=CLASSES, digits=3))

print("🔢 Confusion matrix (rows = true, cols = predicted):")
cm = confusion_matrix(labels, preds)
print(f"  {'':<10} " + " ".join(f"{c[:5]:>6}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    row = " ".join(f"{cm[i,j]:>6d}" for j in range(NUM_CLASSES))
    print(f"  {c:<10} {row}")

# ─── Export to ONNX ─────────────────────────────────────────────────────
print(f"\n📦 Exporting to ONNX...")
model.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
onnx_path = OUT_DIR / f"{MODEL_NAME}.onnx"
torch.onnx.export(
    model,
    dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":  {0: "batch"},
                  "output": {0: "batch"}},
    opset_version=14,
)
print(f"  ✅ {onnx_path}")

# ─── Save metadata ──────────────────────────────────────────────────────
metadata = {
    "model_path":   str(onnx_path),
    "input_size":   IMG_SIZE,
    "input_name":   "input",
    "output_name":  "output",
    "classes":      CLASSES,
    "preprocess":   "imagenet_norm",
    "mean":         [0.485, 0.456, 0.406],
    "std":          [0.229, 0.224, 0.225],
    "input_layout": "NCHW",
    "final_val_accuracy": round(final_acc, 4),
}
with open(OUT_DIR / f"{MODEL_NAME}_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ {OUT_DIR / f'{MODEL_NAME}_meta.json'}")

print(f"\n{'='*60}")
print(f"✅ Training complete!")
print(f"   Best val accuracy: {best_val_acc*100:.2f}%")
print(f"   Model: {onnx_path}")
print(f"   Use this model in emotion_detector.py")
print(f"{'='*60}\n")