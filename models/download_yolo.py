import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# Clear Ultralytics cache to remove any corrupted models
ultralytics_cache = Path.home() / ".ultralytics"
if ultralytics_cache.exists():
    print("🗑️  Clearing Ultralytics cache...")
    shutil.rmtree(ultralytics_cache)

# Download YOLOv8 nano model (official Ultralytics model)
# This model can detect faces and other objects
print("⬇️  Downloading YOLOv8n model from Ultralytics hub...")
model = YOLO("yolov8n.pt")
print("✅ YOLOv8n model ready - saved to:", model.model_name)
