"""Export the already-trained best model to ONNX (no retraining)."""
import torch
import torch.nn as nn
from torchvision import models
import json
from pathlib import Path

# ── Same config as training ──
IMG_SIZE     = 96
NUM_CLASSES  = 7
CLASSES      = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
OUT_DIR      = Path("models")
MODEL_NAME   = "eva_emotion_v1"
DEVICE       = torch.device("cpu")

# ── Rebuild same architecture ──
print("🏗️  Rebuilding model architecture...")
model = models.mobilenet_v2(weights=None)  # no pretrained needed
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES),
)

# ── Load trained weights ──
print(f"📂 Loading weights from {OUT_DIR / f'{MODEL_NAME}_best.pt'}")
model.load_state_dict(torch.load(OUT_DIR / f"{MODEL_NAME}_best.pt", map_location=DEVICE))
model.eval()

# ── Export to ONNX ──
print("📦 Exporting to ONNX...")
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
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

# ── Save metadata ──
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
    "final_val_accuracy": 0.4795,
}
with open(OUT_DIR / f"{MODEL_NAME}_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ {OUT_DIR / f'{MODEL_NAME}_meta.json'}")

print("\n✅ Done! Model ready to use in emotion_detector.py")