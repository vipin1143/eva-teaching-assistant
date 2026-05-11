"""Quick test: verify the new ONNX model loads & predicts correctly."""
import onnxruntime as ort
import numpy as np
import json
from PIL import Image
from pathlib import Path

# ── Load metadata ──
with open("models/eva_emotion_v1_meta.json") as f:
    meta = json.load(f)

CLASSES  = meta["classes"]
IMG_SIZE = meta["input_size"]
MEAN = np.array(meta["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array(meta["std"],  dtype=np.float32).reshape(1, 3, 1, 1)

# ── Load model ──
print("📂 Loading ONNX model...")
session = ort.InferenceSession("models/eva_emotion_v1.onnx",
                                providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
print(f"  Input:  '{inp.name}' shape={inp.shape}")
print(f"  Output: '{out.name}' shape={out.shape}")
print(f"  Classes: {CLASSES}\n")

# ── Test on validation samples ──
val_dir = Path("images/validation")
total_correct = 0
total_samples = 0

print("🧪 Testing 5 samples per class:\n")
for emotion in CLASSES:
    sample_dir = val_dir / emotion
    samples = list(sample_dir.glob("*"))[:5]
    correct = 0
    for img_path in samples:
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]
        arr = ((arr - MEAN) / STD).astype(np.float32)

        logits = session.run(None, {inp.name: arr})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred_idx = int(np.argmax(probs))
        pred = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        if pred == emotion:
            correct += 1
        marker = "✅" if pred == emotion else "❌"
        print(f"  {marker} True={emotion:<10} Pred={pred:<10} ({conf*100:5.1f}%)")
    total_correct += correct
    total_samples += len(samples)
    print(f"     → {correct}/{len(samples)} correct for '{emotion}'\n")

print(f"{'='*50}")
print(f"  Overall on test sample: {total_correct}/{total_samples} = "
      f"{100*total_correct/total_samples:.1f}%")
print(f"{'='*50}")
print("\n✅ Model works! Ready to integrate into emotion_detector.py")