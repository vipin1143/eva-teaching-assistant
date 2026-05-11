"""
models/train_emotion_model.py
─────────────────────────────
Train an emotion-classification model on EVA's real dataset using Ultralytics YOLOv8
classification models.

Why this exists:
- The current EVA project only evaluates a pretrained FER+ ONNX model.
- This script adds REAL supervised training on your own emotion dataset.
- It uses YOLOv8 classification weights (e.g. yolov8n-cls.pt), not the object
  detector weights (yolov8n.pt).

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
    validation/   or   val/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/

Example:
  python models/train_emotion_model.py --data images --model yolov8n-cls.pt --epochs 50 --imgsz 96
  python models/train_emotion_model.py --data images --model yolov8s-cls.pt --epochs 100 --imgsz 128 --batch 32

Outputs:
  runs/emotion-cls/<run_name>/
    weights/best.pt
    weights/last.pt
    results.csv
    results.png
    args.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict


EXPECTED_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_split(split_dir: Path) -> Dict[str, int]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_counts: Dict[str, int] = OrderedDict()
    for class_name in sorted([p.name for p in split_dir.iterdir() if p.is_dir()]):
        class_counts[class_name] = count_images(split_dir / class_name)

    return class_counts


def resolve_validation_dir(data_root: Path) -> Path:
    val_dir = data_root / "val"
    validation_dir = data_root / "validation"

    if val_dir.exists():
        return val_dir
    if validation_dir.exists():
        return validation_dir

    raise FileNotFoundError(
        f"Validation folder not found under {data_root}. Expected either:\n"
        f"  - {val_dir}\n"
        f"  - {validation_dir}"
    )


def validate_dataset(data_root: Path) -> Dict[str, object]:
    train_dir = data_root / "train"
    validation_dir = resolve_validation_dir(data_root)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    train_counts = inspect_split(train_dir)
    val_counts = inspect_split(validation_dir)

    train_classes = list(train_counts.keys())
    val_classes = list(val_counts.keys())

    if train_classes != val_classes:
        raise ValueError(
            "Train/validation class folders do not match.\n"
            f"train: {train_classes}\n"
            f"validation: {val_classes}"
        )

    missing_expected = [c for c in EXPECTED_CLASSES if c not in train_counts]
    extra_classes = [c for c in train_counts if c not in EXPECTED_CLASSES]

    empty_train = [name for name, count in train_counts.items() if count == 0]
    empty_val = [name for name, count in val_counts.items() if count == 0]

    if empty_train or empty_val:
        raise ValueError(
            "Some classes have zero images.\n"
            f"empty train classes: {empty_train}\n"
            f"empty validation classes: {empty_val}"
        )

    return {
        "train": train_counts,
        "validation": val_counts,
        "train_dir": str(train_dir),
        "validation_dir": str(validation_dir),
        "missing_expected": missing_expected,
        "extra_classes": extra_classes,
    }


def print_dataset_summary(dataset_info: Dict[str, object]) -> None:
    print("=" * 72)
    print("EVA EMOTION DATASET SUMMARY")
    print("=" * 72)

    print(f"\nTrain folder      : {dataset_info['train_dir']}")
    print(f"Validation folder : {dataset_info['validation_dir']}")

    print("\nTrain split:")
    total_train = 0
    for class_name, count in dataset_info["train"].items():
        total_train += count
        print(f"  {class_name:<12} {count:>6} images")
    print(f"  {'TOTAL':<12} {total_train:>6} images")

    print("\nValidation split:")
    total_val = 0
    for class_name, count in dataset_info["validation"].items():
        total_val += count
        print(f"  {class_name:<12} {count:>6} images")
    print(f"  {'TOTAL':<12} {total_val:>6} images")

    if dataset_info["missing_expected"]:
        print(f"\n⚠️ Missing expected classes: {dataset_info['missing_expected']}")
    if dataset_info["extra_classes"]:
        print(f"⚠️ Extra classes not in EVA standard mapping: {dataset_info['extra_classes']}")

    print("\nRecommended notes:")
    print("- Use yolov8n-cls.pt for faster training")
    print("- Use yolov8s-cls.pt for potentially better accuracy")
    print("- Start with imgsz=96 or 128 for facial emotion classification")
    print("- Track per-class performance, not just overall accuracy")
    print("=" * 72)


def save_dataset_summary(dataset_info: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    return summary_path


def mirror_split(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_ultralytics_dataset(data_root: Path, run_dir: Path) -> Path:
    """
    Ultralytics classification expects train/ and usually val/.
    EVA currently stores validation data under validation/, so this function
    creates a prepared dataset view with train/ and val/ when needed.
    """
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    validation_dir = data_root / "validation"

    if val_dir.exists():
        return data_root

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Neither '{val_dir}' nor '{validation_dir}' exists. Cannot prepare dataset."
        )

    prepared_root = run_dir / "prepared_dataset"
    prepared_train = prepared_root / "train"
    prepared_val = prepared_root / "val"

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    print("\nℹ️ Preparing Ultralytics-compatible dataset view...")
    print(f"   source train      : {train_dir}")
    print(f"   source validation : {validation_dir}")
    print(f"   prepared root     : {prepared_root}")

    mirror_split(train_dir, prepared_train)
    mirror_split(validation_dir, prepared_val)

    return prepared_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EVA emotion classifier with YOLOv8 classification weights."
    )
    parser.add_argument(
        "--data",
        default="images",
        help="Dataset root containing train/ and validation/ or val/ folders (default: images)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Ultralytics classification model checkpoint to start from",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=96, help="Input image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device, e.g. cpu, 0, 0,1",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument(
        "--project",
        default="runs/emotion-cls",
        help="Directory where Ultralytics stores training runs",
    )
    parser.add_argument(
        "--name",
        default="yolov8n-emotion",
        help="Run name under the project directory",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Classifier dropout to reduce overfitting",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate dataset and prepare Ultralytics-compatible folders without starting training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data).resolve()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install with:\n"
            "  pip install ultralytics"
        ) from exc

    dataset_info = validate_dataset(data_root)
    print_dataset_summary(dataset_info)

    run_dir = Path(args.project) / args.name
    summary_path = save_dataset_summary(dataset_info, run_dir)
    ultralytics_data_root = prepare_ultralytics_dataset(data_root, run_dir)

    print(f"\n✅ Dataset summary saved: {summary_path}")
    print(f"✅ Training data root   : {ultralytics_data_root}")

    if args.inspect_only:
        print("\n✅ Inspection complete. Training was not started because --inspect-only was used.")
        return

    print("\n🚀 Starting training...")
    print(f"   model   : {args.model}")
    print(f"   data    : {ultralytics_data_root}")
    print(f"   epochs  : {args.epochs}")
    print(f"   imgsz   : {args.imgsz}")
    print(f"   batch   : {args.batch}")
    print(f"   device  : {args.device}")
    print(f"   patience: {args.patience}")
    print(f"   project : {args.project}")
    print(f"   name    : {args.name}")

    model = YOLO(args.model)
    results = model.train(
        data=str(ultralytics_data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        dropout=args.dropout,
        lr0=args.lr0,
        pretrained=True,
        verbose=True,
        seed=42,
        exist_ok=True,
    )

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_path = Path(args.project) / args.name / "weights" / "last.pt"

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Best weights: {best_path}")
    print(f"Last weights: {last_path}")
    print(f"Ultralytics results object: {results}")
    print("\nNext step:")
    print(
        f"  python evaluation/evaluate_yolo_classifier.py --model \"{best_path}\" --data \"{data_root}\""
    )


if __name__ == "__main__":
    main()
