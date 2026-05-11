"""
evaluation/evaluate_yolo_classifier.py
──────────────────────────────────────
Evaluate a trained YOLOv8 classification model on EVA's validation dataset.

This script is for the NEW trainable emotion classifier workflow, not the old FER+ ONNX
pipeline. It computes:
- overall accuracy
- per-class precision / recall / F1
- confusion matrix
- class counts
- optional matplotlib heatmap
- saved text + JSON reports

Expected dataset layout:
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

Example:
  python evaluation/evaluate_yolo_classifier.py --model runs/emotion-cls/yolov8n-emotion/weights/best.pt --data images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EXPECTED_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EVA YOLOv8 emotion classifier")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 classification model (.pt)")
    parser.add_argument(
        "--data",
        default="images",
        help="Dataset root containing train/ and validation/ folders (default: images)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "train"],
        help="Which dataset split to evaluate",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=96,
        help="Inference image size used during evaluation",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, e.g. cpu, 0, 0,1",
    )
    parser.add_argument(
        "--output",
        default="runs/emotion-eval",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--name",
        default="latest",
        help="Subdirectory name under the output directory",
    )
    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        help="Save confusion matrix heatmap PNG",
    )
    return parser.parse_args()


def list_class_dirs(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def validate_dataset(data_root: Path, split: str) -> Tuple[Path, List[str]]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_dirs = list_class_dirs(split_dir)
    if not class_dirs:
        raise ValueError(f"No class folders found in: {split_dir}")

    class_names = [p.name for p in class_dirs]

    empty_classes = [p.name for p in class_dirs if not list_images(p)]
    if empty_classes:
        raise ValueError(f"These classes contain zero images: {empty_classes}")

    return split_dir, class_names


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]):
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
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(conf_matrix[i, :].sum())

        metrics[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": support,
            "correct": int(tp),
        }

    total_correct = int(sum(conf_matrix[i][i] for i in range(n_classes)))
    total_samples = int(conf_matrix.sum())
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

    return round(float(accuracy), 4), metrics, conf_matrix


def print_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str]) -> None:
    print("\n" + "=" * 78)
    print("CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print("=" * 78)

    col_w = max(8, max(len(c) for c in class_names) + 1)
    header = "actual \\ pred".ljust(col_w + 4)
    for name in class_names:
        header += name[:col_w].center(col_w)
    print(header)
    print("-" * len(header))

    for i, row_name in enumerate(class_names):
        row = row_name.ljust(col_w + 4)
        for j in range(len(class_names)):
            row += str(conf_matrix[i][j]).center(col_w)
        print(row)


def print_report(
    accuracy: float,
    metrics: Dict[str, Dict[str, float]],
    class_names: List[str],
    avg_confidence: float,
) -> None:
    print("\n" + "=" * 78)
    print("EVA YOLOv8 EMOTION CLASSIFIER REPORT")
    print("=" * 78)
    print(f"Overall Accuracy   : {accuracy * 100:.2f}%")
    print(f"Average Confidence : {avg_confidence * 100:.2f}%")
    print()

    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 58)
    for cls in class_names:
        m = metrics[cls]
        print(
            f"{cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} {m['support']:>10}"
        )

    macro_p = np.mean([metrics[c]["precision"] for c in class_names])
    macro_r = np.mean([metrics[c]["recall"] for c in class_names])
    macro_f1 = np.mean([metrics[c]["f1_score"] for c in class_names])

    print("-" * 58)
    print(f"{'Macro Avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")


def save_heatmap(conf_matrix: np.ndarray, class_names: List[str], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        conf_matrix.astype(float),
        row_sums,
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalized proportion")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = norm_cm.max() / 2 if norm_cm.size else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = conf_matrix[i][j]
            pct = norm_cm[i][j]
            color = "white" if pct > threshold else "black"
            ax.text(j, i, f"{count}\n({pct * 100:.0f}%)", ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("EVA Emotion Classifier Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_reports(
    output_dir: Path,
    accuracy: float,
    avg_confidence: float,
    metrics: Dict[str, Dict[str, float]],
    conf_matrix: np.ndarray,
    class_names: List[str],
    samples: List[Dict[str, object]],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "report.json"
    txt_path = output_dir / "report.txt"

    report_data = {
        "accuracy": accuracy,
        "average_confidence": round(avg_confidence, 4),
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "sample_count": len(samples),
        "samples": samples,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("EVA YOLOv8 Emotion Classifier Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Overall Accuracy   : {accuracy * 100:.2f}%\n")
        f.write(f"Average Confidence : {avg_confidence * 100:.2f}%\n\n")
        f.write(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-" * 58 + "\n")
        for cls in class_names:
            m = metrics[cls]
            f.write(
                f"{cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} {m['support']:>10}\n"
            )
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

    return json_path, txt_path


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install with:\n"
            "  pip install ultralytics"
        ) from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    data_root = Path(args.data).resolve()
    split_dir, class_names = validate_dataset(data_root, args.split)

    if sorted(class_names) != sorted(EXPECTED_CLASSES):
        print(f"⚠️ Dataset classes differ from EVA standard classes: {class_names}")

    print("=" * 78)
    print("EVA YOLOv8 CLASSIFIER EVALUATION")
    print("=" * 78)
    print(f"Model : {model_path}")
    print(f"Data  : {split_dir}")
    print(f"Split : {args.split}")
    print(f"Device: {args.device}")
    print(f"Imgsz : {args.imgsz}")

    model = YOLO(str(model_path))
    model_names = model.names
    if isinstance(model_names, dict):
        ordered_model_names = [model_names[i] for i in sorted(model_names)]
    else:
        ordered_model_names = list(model_names)

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    model_class_to_index = {name: idx for idx, name in enumerate(ordered_model_names)}

    missing_in_model = [name for name in class_names if name not in model_class_to_index]
    if missing_in_model:
        raise SystemExit(
            "Model class names do not match dataset classes.\n"
            f"Missing in model: {missing_in_model}\n"
            f"Model classes: {ordered_model_names}"
        )

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    samples: List[Dict[str, object]] = []

    for class_dir in list_class_dirs(split_dir):
        true_label = class_dir.name
        true_idx = class_to_index[true_label]
        image_paths = list_images(class_dir)
        print(f"\n📂 {true_label:<12} {len(image_paths):>5} images")

        for image_path in image_paths:
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            if not results:
                continue

            probs = results[0].probs
            if probs is None:
                continue

            pred_idx_model = int(probs.top1)
            pred_name = ordered_model_names[pred_idx_model]
            pred_idx_dataset = class_to_index[pred_name]
            confidence = float(probs.top1conf)

            y_true.append(true_idx)
            y_pred.append(pred_idx_dataset)
            confidences.append(confidence)

            samples.append(
                {
                    "image": str(image_path),
                    "true_label": true_label,
                    "pred_label": pred_name,
                    "confidence": round(confidence, 4),
                }
            )

    if not y_true:
        raise SystemExit("No samples were evaluated. Check the dataset path and images.")

    accuracy, metrics, conf_matrix = compute_metrics(y_true, y_pred, class_names)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    print_confusion_matrix(conf_matrix, class_names)
    print_report(accuracy, metrics, class_names, avg_confidence)

    output_dir = Path(args.output) / args.name
    json_path, txt_path = save_reports(
        output_dir=output_dir,
        accuracy=accuracy,
        avg_confidence=avg_confidence,
        metrics=metrics,
        conf_matrix=conf_matrix,
        class_names=class_names,
        samples=samples,
    )

    print(f"\n✅ JSON report saved: {json_path}")
    print(f"✅ Text report saved: {txt_path}")

    if args.save_heatmap:
        heatmap_path = output_dir / "confusion_matrix.png"
        save_heatmap(conf_matrix, class_names, heatmap_path)
        print(f"✅ Heatmap saved: {heatmap_path}")

    print("\n" + "=" * 78)
    print(f"FINAL RESULT: Accuracy = {accuracy * 100:.2f}% on {len(y_true)} samples")
    print("=" * 78)


if __name__ == "__main__":
    main()
