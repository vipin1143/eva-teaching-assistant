"""
evaluate_model.py
─────────────────
Improvement 3: Accuracy Metrics

Generates:
  - Per-class accuracy
  - Precision / Recall / F1-Score  
  - Confusion matrix (text + matplotlib heatmap)
  - Overall accuracy

Run this script to evaluate the FER+ ONNX model on sample test images.

Usage:
  python evaluate_model.py                     # synthetic test (no data needed)
  python evaluate_model.py --data /path/to/fer # if you have FER dataset

What the metrics mean:
  Accuracy  = correctly predicted / total
  Precision = when EVA says "confused", how often it's right
  Recall    = of all confused students, how many EVA caught
  F1        = balance between precision and recall
"""

import os, sys, argparse, logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

FERPLUS_EMOTIONS = ["neutral","happiness","surprise","sadness",
                    "anger","disgust","fear","contempt"]

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
    """
    Run FER+ model on a preprocessed 64x64 grayscale image.
    Returns predicted class index and confidence.
    """
    inp_name = session.get_inputs()[0].name
    inp      = img_array.reshape(1, 1, 64, 64).astype(np.float32)
    scores   = session.run(None, {inp_name: inp})[0][0]
    exp      = np.exp(scores - np.max(scores))
    probs    = exp / exp.sum()
    return int(np.argmax(probs)), float(np.max(probs)), probs


def preprocess_image(img_path: str):
    """Load and preprocess a face image for the FER+ model."""
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Same preprocessing as emotion_detector.py
    img = cv2.resize(img, (64, 64))
    img = cv2.equalizeHist(img)
    return img.astype(np.float32)


def evaluate_on_folder(session, data_dir: str):
    """
    Evaluate model on a folder structure:
      data_dir/
        neutral/   img1.jpg, img2.jpg ...
        happiness/ img1.jpg ...
        sadness/   ...
    """
    import cv2
    y_true, y_pred = [], []

    for class_idx, emotion in enumerate(FERPLUS_EMOTIONS):
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            # Try common alternative names
            alt = {"happiness":"happy","sadness":"sad","anger":"angry","contempt":"frustrated"}
            folder = os.path.join(data_dir, alt.get(emotion, emotion))

        if not os.path.exists(folder):
            print(f"  ⚠️  Folder not found: {emotion} — skipping")
            continue

        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))][:100]

        print(f"  📂 {emotion}: {len(images)} images")

        for fname in images:
            img = preprocess_image(os.path.join(folder, fname))
            if img is None:
                continue
            pred_idx, _, _ = run_onnx_on_image(session, img)
            y_true.append(class_idx)
            y_pred.append(pred_idx)

    return y_true, y_pred


def evaluate_synthetic(session):
    """
    Synthetic evaluation when no real dataset is available.
    Creates realistic test images using pixel patterns that approximate
    different facial expressions (for demonstration purposes).
    """
    import cv2
    print("📊 Running synthetic evaluation (no dataset needed)")
    print("   Note: For real accuracy numbers, provide --data /path/to/FER+")
    print()

    y_true, y_pred, confidences = [], [], []

    # Generate 50 test samples per class using controlled patterns
    np.random.seed(42)
    for true_class_idx in range(8):
        for sample_idx in range(50):
            # Create a base face pattern (64x64)
            face = np.zeros((64, 64), dtype=np.float32)

            # Draw a rough face structure
            # Head oval
            center = (32, 32)
            for y in range(64):
                for x in range(64):
                    # Ellipse for face region
                    if ((x-32)/22)**2 + ((y-30)/26)**2 < 1:
                        face[y,x] = 150 + np.random.normal(0, 20)

            # Add emotion-specific patterns
            emotion = FERPLUS_EMOTIONS[true_class_idx]

            if emotion == "happiness":
                # Smile curve (higher pixel values in mouth area)
                for x in range(20, 44):
                    y_smile = int(48 + 5 * np.sin((x - 32) * 0.3))
                    if 0 <= y_smile < 64:
                        face[y_smile, x] = min(255, face[y_smile,x] + 80)

            elif emotion == "sadness":
                # Drooping expression
                face[38:45, 18:26] = np.clip(face[38:45,18:26] - 40, 0, 255)
                face[38:45, 38:46] = np.clip(face[38:45,38:46] - 40, 0, 255)

            elif emotion == "anger":
                # Furrowed brows
                face[20:24, 22:32] = np.clip(face[20:24,22:32] + 60, 0, 255)
                face[20:24, 32:42] = np.clip(face[20:24,32:42] + 60, 0, 255)

            elif emotion == "surprise":
                # Wide eyes + open mouth
                face[22:28, 20:28] = 250
                face[22:28, 36:44] = 250
                face[44:52, 24:40] = 50

            # Normalize and add noise
            face = np.clip(face + np.random.normal(0, 15, face.shape), 0, 255)
            face = face.astype(np.float32)

            pred_idx, conf, _ = run_onnx_on_image(session, face)
            y_true.append(true_class_idx)
            y_pred.append(pred_idx)
            confidences.append(conf)

    return y_true, y_pred, confidences


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute per-class and overall metrics.
    Returns: accuracy, per_class_metrics, confusion_matrix
    """
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
        tn = conf_matrix.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = conf_matrix[i, :].sum()

        metrics[cls] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1_score":  round(f1, 3),
            "support":   int(support),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)
        }

    total_correct = sum(conf_matrix[i][i] for i in range(n_classes))
    total_samples = conf_matrix.sum()
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return round(accuracy, 4), metrics, conf_matrix


def print_confusion_matrix(conf_matrix, class_names):
    """Print confusion matrix as ASCII table."""
    max_name = max(len(n) for n in class_names)
    col_w    = max(5, max_name)

    print("\n" + "─" * 70)
    print("CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print("─" * 70)

    # Header
    header = " " * (max_name + 2)
    for n in class_names:
        header += f"{n[:7]:^8}"
    print(header)
    print(" " * (max_name + 2) + "─" * (8 * len(class_names)))

    for i, row_name in enumerate(class_names):
        row = f"{row_name:<{max_name}} |"
        for j in range(len(class_names)):
            val = conf_matrix[i][j]
            if i == j:
                row += f"\033[92m{val:^8}\033[0m"  # green on diagonal
            elif val > 0:
                row += f"\033[93m{val:^8}\033[0m"  # yellow for errors
            else:
                row += f"{val:^8}"
        print(row)

    print("─" * 70)
    print("  ✅ Green = correct predictions, 🟡 Yellow = misclassifications")


def print_classification_report(accuracy, metrics, class_names, confidences=None):
    """Print full classification report."""
    print("\n" + "═" * 70)
    print("EVA EMOTION MODEL — ACCURACY REPORT")
    print("═" * 70)
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")

    if confidences:
        print(f"📊 Average Confidence: {np.mean(confidences)*100:.1f}%")

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("─" * 55)

    for cls in class_names:
        m = metrics[cls]
        display = EMOTION_MAP.get(cls, cls)
        print(f"{display:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1_score']:>10.3f} {m['support']:>10}")

    print("─" * 55)

    avg_p  = np.mean([m["precision"] for m in metrics.values()])
    avg_r  = np.mean([m["recall"]    for m in metrics.values()])
    avg_f1 = np.mean([m["f1_score"]  for m in metrics.values()])
    total  = sum(m["support"] for m in metrics.values())
    print(f"{'Avg/Total':<12} {avg_p:>10.3f} {avg_r:>10.3f} {avg_f1:>10.3f} {total:>10}")
    print("═" * 70)


def save_matplotlib_heatmap(conf_matrix, class_names, output_path="confusion_matrix.png"):
    """Save confusion matrix as a color heatmap PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize confusion matrix to percentages
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        norm_cm  = np.divide(conf_matrix.astype(float), row_sums,
                             out=np.zeros_like(conf_matrix, dtype=float),
                             where=row_sums != 0)

        im = ax.imshow(norm_cm, interpolation='nearest',
                       cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Proportion')

        display_names = [EMOTION_MAP.get(c, c) for c in class_names]
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(display_names, fontsize=11)

        # Add count annotations
        thresh = norm_cm.max() / 2
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = conf_matrix[i][j]
                pct   = norm_cm[i][j]
                color = "white" if pct > thresh else "black"
                text  = f"{count}\n({pct*100:.0f}%)"
                ax.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold' if i==j else 'normal')

        ax.set_ylabel('Actual Emotion', fontsize=13)
        ax.set_xlabel('Predicted Emotion', fontsize=13)
        ax.set_title('EVA — FER+ Model Confusion Matrix\n(diagonal = correct predictions)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"\n✅ Confusion matrix heatmap saved: {output_path}")
        return True
    except ImportError:
        print("ℹ️  matplotlib not installed — skipping heatmap")
        print("   Install: pip install matplotlib")
        return False
    except Exception as e:
        print(f"⚠️  Could not save heatmap: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="EVA Model Accuracy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py                          # synthetic test
  python evaluate_model.py --data ./fer_data        # real FER+ dataset
  python evaluate_model.py --no-plot                # skip matplotlib chart

Output files:
  confusion_matrix.png     — color heatmap of confusion matrix
  accuracy_report.txt      — full text report
        """
    )
    parser.add_argument("--model", default=None, help="Path to ONNX model")
    parser.add_argument("--data",  default=None, help="Path to test dataset folder")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib heatmap")
    parser.add_argument("--output", default=".", help="Output directory for reports")
    args = parser.parse_args()

    # ── Find model ────────────────────────────────────────────
    model_path = args.model
    if not model_path:
        candidates = [
            "models/emotion_ferplus_8.onnx",
            "../models/emotion_ferplus_8.onnx",
            os.path.join(os.path.dirname(__file__), "models/emotion_ferplus_8.onnx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not model_path or not os.path.exists(model_path):
        print("❌ ONNX model not found.")
        print("   Run first: python models/download_models.py")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        print(f"✅ Model loaded: {model_path}")
    except ImportError:
        print("❌ onnxruntime not installed: pip install onnxruntime")
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────
    confidences = None
    if args.data and os.path.exists(args.data):
        print(f"\n📂 Evaluating on dataset: {args.data}")
        y_true, y_pred = evaluate_on_folder(session, args.data)
    else:
        if args.data:
            print(f"⚠️  Dataset path not found: {args.data}")
        y_true, y_pred, confidences = evaluate_synthetic(session)

    if not y_true:
        print("❌ No test samples found.")
        sys.exit(1)

    print(f"\n📊 Total samples evaluated: {len(y_true)}")

    # ── Compute metrics ───────────────────────────────────────
    accuracy, metrics, conf_matrix = compute_metrics(y_true, y_pred, FERPLUS_EMOTIONS)

    # ── Print report ──────────────────────────────────────────
    print_confusion_matrix(conf_matrix, FERPLUS_EMOTIONS)
    print_classification_report(accuracy, metrics, FERPLUS_EMOTIONS, confidences)

    # ── Save reports ──────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # Text report
    report_path = os.path.join(args.output, "accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write("EVA — FER+ ONNX Model Accuracy Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("-" * 45 + "\n")
        for cls in FERPLUS_EMOTIONS:
            m = metrics[cls]
            f.write(f"{EMOTION_MAP.get(cls,cls):<12} {m['precision']:>10.3f} "
                    f"{m['recall']:>10.3f} {m['f1_score']:>10.3f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    print(f"✅ Text report saved: {report_path}")

    # Heatmap
    if not args.no_plot:
        heatmap_path = os.path.join(args.output, "confusion_matrix.png")
        save_matplotlib_heatmap(conf_matrix, FERPLUS_EMOTIONS, heatmap_path)

    print(f"\n{'='*70}")
    print(f"  FINAL RESULT: Overall Accuracy = {accuracy*100:.2f}%")
    print(f"{'='*70}\n")

    return accuracy


if __name__ == "__main__":
    main()