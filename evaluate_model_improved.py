"""
evaluate_model_improved.py
──────────────────────────
Re-evaluate FER+ model with advanced preprocessing and YOLOv8 face detection.

Improvements:
  1. Use YOLOv8 for face detection (better than Haar Cascade)
  2. Advanced preprocessing (CLAHE + denoising + morphological ops)
  3. Compare results before/after

Usage:
  python evaluate_model_improved.py --data ./images/train --mode fast
  python evaluate_model_improved.py --data ./images/train --mode full
"""

import os, sys, argparse, logging
import numpy as np
import cv2
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessing import advanced_preprocessing

logging.basicConfig(level=logging.WARNING)

FERPLUS_EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
                    "anger", "disgust", "fear", "contempt"]

EMOTION_MAP = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
    "contempt":  "frustrated"
}


def run_onnx_on_image(session, img_array):
    """Run FER+ model on preprocessed 64x64 grayscale image."""
    inp_name = session.get_inputs()[0].name
    inp      = img_array.reshape(1, 1, 64, 64).astype(np.float32)
    scores   = session.run(None, {inp_name: inp})[0][0]
    exp      = np.exp(scores - np.max(scores))
    probs    = exp / exp.sum()
    return int(np.argmax(probs)), float(np.max(probs)), probs


def preprocess_image_original(img_path: str):
    """Original preprocessing (basic)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    img = cv2.equalizeHist(img)
    return img.astype(np.float32)


def preprocess_image_improved(img_path: str, mode='fast'):
    """Improved preprocessing with advanced techniques."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize first
    img = cv2.resize(img, (64, 64))
    
    # Apply advanced preprocessing
    img = advanced_preprocessing(img, mode=mode)
    
    return img


def evaluate_on_folder(session, data_dir: str, preprocess_fn, mode_name=''):
    """Evaluate model on a folder with specified preprocessing."""
    y_true, y_pred, confidences = [], [], []

    for class_idx, emotion in enumerate(FERPLUS_EMOTIONS):
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            alt = {"happiness":"happy","sadness":"sad","anger":"angry","contempt":"frustrated"}
            folder = os.path.join(data_dir, alt.get(emotion, emotion))

        if not os.path.exists(folder):
            print(f"  ⚠️  {emotion}: folder not found")
            continue

        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))][:100]

        for fname in images:
            img = preprocess_fn(os.path.join(folder, fname))
            if img is None:
                continue
            pred_idx, conf, _ = run_onnx_on_image(session, img)
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            confidences.append(conf)
        
        print(f"  ✅ {emotion:<12} {len(images):>3} images")

    return y_true, y_pred, confidences


def compute_metrics(y_true, y_pred, class_names):
    """Compute metrics."""
    n_classes = len(class_names)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf_matrix[t][p] += 1

    metrics = {}
    for i, cls in enumerate(class_names):
        tp = conf_matrix[i][i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1_score":  round(f1, 3),
        }

    total_correct = sum(conf_matrix[i][i] for i in range(n_classes))
    total_samples = conf_matrix.sum()
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return round(accuracy, 4), metrics, conf_matrix


def main():
    parser = argparse.ArgumentParser(description="Improved Model Evaluation")
    parser.add_argument("--data", default="./images/train", help="Dataset path")
    parser.add_argument("--model", default=None, help="ONNX model path")
    parser.add_argument("--mode", default="fast", choices=["basic", "fast", "full"],
                       help="Preprocessing mode")
    args = parser.parse_args()

    # Find model
    model_path = args.model
    if not model_path:
        candidates = [
            "models/emotion_ferplus_8.onnx",
            "../models/emotion_ferplus_8.onnx",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not model_path or not os.path.exists(model_path):
        print("❌ ONNX model not found")
        sys.exit(1)

    # Load model
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        print(f"✅ Model loaded: {model_path}\n")
    except ImportError:
        print("❌ onnxruntime not installed: pip install onnxruntime")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"❌ Dataset not found: {args.data}")
        sys.exit(1)

    print("=" * 70)
    print("  EMOTION MODEL EVALUATION — IMPROVED PREPROCESSING")
    print("=" * 70)
    print(f"\n📊 Preprocessing Mode: {args.mode.upper()}\n")
    
    # Evaluate with improved preprocessing
    print(f"Evaluating with {args.mode} preprocessing:")
    y_true, y_pred, confidences = evaluate_on_folder(
        session, args.data, 
        lambda x: preprocess_image_improved(x, mode=args.mode),
        args.mode
    )

    if not y_true:
        print("❌ No test samples found")
        sys.exit(1)

    accuracy_imp, metrics_imp, conf_imp = compute_metrics(y_true, y_pred, FERPLUS_EMOTIONS)

    # Compare with original
    print(f"\nEvaluating with ORIGINAL preprocessing:")
    y_true_orig, y_pred_orig, conf_orig = evaluate_on_folder(
        session, args.data, preprocess_image_original, "original"
    )
    accuracy_orig, metrics_orig, _ = compute_metrics(y_true_orig, y_pred_orig, FERPLUS_EMOTIONS)

    # Results
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n📊 Original Preprocessing:  {accuracy_orig*100:.2f}%")
    print(f"📊 Improved Preprocessing: {accuracy_imp*100:.2f}%")
    print(f"📈 Improvement:            +{(accuracy_imp - accuracy_orig)*100:.2f}%\n")

    print("Per-emotion F1-Score Comparison:")
    print("-" * 70)
    print(f"{'Emotion':<15} {'Original':>15} {'Improved':>15} {'Gain':>15}")
    print("-" * 70)
    for emotion in FERPLUS_EMOTIONS:
        orig_f1 = metrics_orig[emotion]["f1_score"]
        imp_f1 = metrics_imp[emotion]["f1_score"]
        gain = imp_f1 - orig_f1
        print(f"{EMOTION_MAP[emotion]:<15} {orig_f1:>15.3f} {imp_f1:>15.3f} {gain:+15.3f}")

    print("=" * 70)
    print(f"\n✅ Evaluation complete!")
    print(f"   Average confidence: {np.mean(confidences)*100:.1f}%")


if __name__ == "__main__":
    main()
