# EVA Emotion Model Training Guide

This guide explains how to train **your own facial emotion model** on the real dataset already present in this project.

## What changed

The original project only used:

- a **pretrained FER+ ONNX** model for emotion classification
- **YOLO** for face detection

This repository now also includes a **real training pipeline** for a new emotion classifier using **Ultralytics YOLOv8 classification models**.

New files added:

- `models/train_emotion_model.py`
- `evaluation/evaluate_yolo_classifier.py`

---

## Important concept

There are **two different YOLO use cases** in this project:

### 1. Face detection
Used to find the face in webcam/classroom frames.

Examples:
- `yolov8n-face`
- `yolov8m.pt` in current detector code

### 2. Emotion classification
Used to predict one label from a cropped face image:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

For this, you should use a **classification model**, not plain object detection weights.

Use:
- `yolov8n-cls.pt` for faster training
- `yolov8s-cls.pt` for potentially better accuracy

Do **not** use plain `yolov8n.pt` for this emotion-classification dataset.

---

## Current dataset

Your dataset is already organized well for classification:

```text
images/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  validation/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

The training script supports both:
- `validation/`
- `val/`

If only `validation/` exists, it automatically creates a temporary Ultralytics-compatible `val/` view inside the run folder.

---

## Dataset counts detected

From the current project:

### Train
- angry: 3993
- disgust: 436
- fear: 4103
- happy: 7164
- neutral: 4982
- sad: 4938
- surprise: 3205

### Validation
- angry: 960
- disgust: 111
- fear: 1018
- happy: 1825
- neutral: 1216
- sad: 1139
- surprise: 797

## Important imbalance note

Your `disgust` class is much smaller than the others.

That means the model may get:
- high overall accuracy
- but weak recall/F1 for `disgust`

When evaluating, pay attention to:
- per-class recall
- per-class F1
- confusion matrix

Not just overall accuracy.

---

## Step 1 — Inspect dataset only

This checks:
- class folders
- image counts
- train/validation consistency
- Ultralytics `val/` preparation logic

Run:

```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --inspect-only
```

---

## Step 2 — Train the first model

Recommended first run:

```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8n-cls.pt --epochs 50 --imgsz 96 --batch 32 --device cpu --name yolov8n-emotion
```

### Recommended stronger run
If training time is acceptable:

```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8s-cls.pt --epochs 100 --imgsz 128 --batch 32 --device cpu --name yolov8s-emotion
```

### If you have GPU
Example:

```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8s-cls.pt --epochs 100 --imgsz 128 --batch 32 --device 0 --name yolov8s-emotion-gpu
```

---

## Step 3 — Evaluate the trained model

After training, evaluate `best.pt` on the validation set.

Example:

```bash
python eva-project/evaluation/evaluate_yolo_classifier.py --model runs/emotion-cls/yolov8n-emotion/weights/best.pt --data eva-project/images --split validation --imgsz 96 --device cpu --name yolov8n-eval --save-heatmap
```

For a `yolov8s` run:

```bash
python eva-project/evaluation/evaluate_yolo_classifier.py --model runs/emotion-cls/yolov8s-emotion/weights/best.pt --data eva-project/images --split validation --imgsz 128 --device cpu --name yolov8s-eval --save-heatmap
```

Outputs are saved under:

```text
runs/emotion-eval/<name>/
  report.json
  report.txt
  confusion_matrix.png   # if --save-heatmap used
```

---

## Step 4 — Compare models

Train at least two runs and compare:

- `yolov8n-cls.pt`, `imgsz=96`
- `yolov8s-cls.pt`, `imgsz=128`

Compare these metrics:
- overall accuracy
- macro F1
- recall for `fear`, `sad`, `disgust`
- confusion matrix

---

## Recommended training progression

### Baseline
```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8n-cls.pt --epochs 50 --imgsz 96 --batch 32 --device cpu --name baseline-n96
```

### Better accuracy
```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8s-cls.pt --epochs 100 --imgsz 128 --batch 32 --device cpu --name stronger-s128
```

### If overfitting appears
Try:
- fewer epochs
- higher dropout
- slightly lower image size
- more balanced data

Example:

```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8s-cls.pt --epochs 60 --imgsz 96 --batch 32 --dropout 0.2 --lr0 0.0005 --device cpu --name regularized-s96
```

---

## How to judge if the model is actually better

A model is better if:

- validation accuracy improves
- macro F1 improves
- `disgust`, `fear`, and `sad` improve
- confidence is not high on wrong predictions
- confusion matrix is less concentrated in wrong nearby classes

Common confusion pairs:
- fear ↔ surprise
- sad ↔ neutral
- angry ↔ disgust

---

## Why your real dataset should improve accuracy

Your old FER+ ONNX model was trained on a general dataset.

Your real data likely differs in:
- webcam quality
- lighting
- student posture
- classroom capture conditions
- compression artifacts
- demographic/domain shift

Training on your own dataset should usually improve performance more than only tweaking preprocessing.

---

## Best practical architecture for EVA

### Recommended production pipeline

1. **YOLO face detector** finds face in frame
2. crop the face
3. send crop to the **trained emotion classifier**
4. combine with head pose if needed
5. log emotion result

That means:

- keep YOLO for detection
- replace the pretrained FER+ classifier with your new trained classifier once it performs better

---

## Current project caveat

`utils/emotion_detector.py` still uses:

- `emotion_ferplus_8.onnx` for emotion classification

So right now the new trained classifier is **trainable and evaluatable**, but not yet wired into the live inference pipeline.

If you want full integration later, the next step is:

- load the trained `best.pt` classifier in `emotion_detector.py`
- run it on cropped faces
- compare its output against the old ONNX model
- then switch production inference to the better model

---

## Recommended next action

Run these in order:

### 1. Inspect
```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --inspect-only
```

### 2. Train baseline
```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8n-cls.pt --epochs 50 --imgsz 96 --batch 32 --device cpu --name yolov8n-emotion
```

### 3. Evaluate
```bash
python eva-project/evaluation/evaluate_yolo_classifier.py --model runs/emotion-cls/yolov8n-emotion/weights/best.pt --data eva-project/images --split validation --imgsz 96 --device cpu --name yolov8n-eval --save-heatmap
```

### 4. Train stronger model
```bash
python eva-project/models/train_emotion_model.py --data eva-project/images --model yolov8s-cls.pt --epochs 100 --imgsz 128 --batch 32 --device cpu --name yolov8s-emotion
```

### 5. Evaluate stronger model
```bash
python eva-project/evaluation/evaluate_yolo_classifier.py --model runs/emotion-cls/yolov8s-emotion/weights/best.pt --data eva-project/images --split validation --imgsz 128 --device cpu --name yolov8s-eval --save-heatmap
```

---

## Summary

Use this rule:

- **YOLO face model** → detect faces
- **YOLO classification model (`*-cls.pt`)** → classify emotions

For your dataset, the best next move is:
1. train `yolov8n-cls` baseline
2. train `yolov8s-cls` stronger version
3. compare validation metrics
4. integrate the winner into `emotion_detector.py`
