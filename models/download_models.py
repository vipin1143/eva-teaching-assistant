"""
models/download_models.py
─────────────────────────
Downloads all required model files for EVA:
  1. emotion_ferplus_8.onnx                  — FER+ emotion model (ONNX)
  2. haarcascade_frontalface_default.xml     — OpenCV face detector
  3. face_landmarker.tflite                  — MediaPipe head pose model

Usage:
    python models/download_models.py           # download all missing
    python models/download_models.py --check   # only check status
    python models/download_models.py --force   # re-download all
"""

import os, sys, argparse, urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "emotion_ferplus_8.onnx": {
        "url": (
            "https://github.com/onnx/models/raw/main/validated/"
            "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
        ),
        "min_size": 30_000_000,
        "description": "FER+ emotion recognition model (8 classes)",
    },
    "haarcascade_frontalface_default.xml": {
        "url": (
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "data/haarcascades/haarcascade_frontalface_default.xml"
        ),
        "min_size": 800_000,
        "description": "OpenCV Haar cascade face detector",
    },
    "face_landmarker.tflite": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        ),
        "min_size": 1_000_000,
        "description": "MediaPipe face landmarker for head pose estimation",
    },
    "yolov8n-face.pt": {
        "url": None,   # downloaded via ultralytics auto-download
        "min_size": 5_000_000,
        "description": "YOLOv8n-face detector — replaces Haar Cascade (95%+ accuracy)",
    },
}


def _bar(done, total, width=40):
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done/1048576:.1f}/{total/1048576:.1f} MB ({pct*100:.0f}%)"


def _file_ok(path, min_size):
    return os.path.isfile(path) and os.path.getsize(path) >= min_size


def download_model(filename, meta, force=False):
    path = os.path.join(MODELS_DIR, filename)

    if not force and _file_ok(path, meta["min_size"]):
        print(f"  ✅  {filename}  ({os.path.getsize(path)/1048576:.1f} MB) — already present")
        return True

    print(f"\n  ⬇️   Downloading: {filename}")
    print(f"       {meta['description']}")

    try:
        # Try requests first (better progress bar)
        try:
            import requests
            r = requests.get(meta["url"], stream=True, timeout=120)
            r.raise_for_status()
            total      = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r       {_bar(downloaded, total)}", end="", flush=True)
            print()
        except ImportError:
            # Fallback to urllib
            print(f"       Source: {meta['url']}")
            urllib.request.urlretrieve(meta["url"], path)

        actual = os.path.getsize(path)
        if actual < meta["min_size"]:
            print(f"  ❌  File too small ({actual/1048576:.1f} MB) — download may have failed")
            os.remove(path)
            return False

        print(f"  ✅  Saved → {path}  ({actual/1048576:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n  ❌  Download failed: {e}")
        if os.path.exists(path):
            os.remove(path)
        return False


def try_opencv_cascade():
    """Copy Haar cascade from OpenCV package if download fails."""
    dest = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
    if os.path.isfile(dest):
        return True
    try:
        import cv2, shutil
        src = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.isfile(src):
            shutil.copy2(src, dest)
            print(f"  ✅  Copied haarcascade from OpenCV package")
            return True
    except Exception:
        pass
    return False


def print_status():
    print("\n  Model Status")
    print("  " + "─" * 58)
    all_ok = True
    for filename, meta in MODELS.items():
        path = os.path.join(MODELS_DIR, filename)
        if _file_ok(path, meta["min_size"]):
            size = os.path.getsize(path) / 1048576
            print(f"  ✅  {filename:<46} {size:>5.1f} MB")
        else:
            print(f"  ❌  {filename:<46}  MISSING")
            all_ok = False
    print("  " + "─" * 58)
    if all_ok:
        print("  All models ready — EVA can start!\n")
    else:
        print("  Run: python models/download_models.py\n")
    return all_ok


def download_yolov8_face():
    """
    Download YOLOv8n-face model using ultralytics auto-download.
    ultralytics handles CDN download automatically.
    """
    dest = os.path.join(MODELS_DIR, "yolov8n-face.pt")
    if os.path.exists(dest) and os.path.getsize(dest) >= 5_000_000:
        print(f"  ✅  yolov8n-face.pt  ({os.path.getsize(dest)/1048576:.1f} MB) — already present")
        return True

    print("  ⬇️   Downloading yolov8n-face.pt via ultralytics...")

    # Method 1: ultralytics auto-download (recommended)
    try:
        from ultralytics import YOLO
        import shutil, glob
        # Load model — ultralytics downloads to ~/.cache/ultralytics/
        model = YOLO("yolov8n-face.pt")
        # Find the downloaded file in cache
        cache_dirs = [
            os.path.expanduser("~/.cache/ultralytics/"),
            os.path.expanduser("~/AppData/Roaming/Ultralytics/"),
            os.path.join(os.getcwd(), "yolov8n-face.pt"),
        ]
        for d in cache_dirs:
            matches = glob.glob(os.path.join(d, "**", "yolov8n-face.pt"), recursive=True)
            if matches:
                shutil.copy2(matches[0], dest)
                print(f"  ✅  yolov8n-face.pt saved ({os.path.getsize(dest)/1048576:.1f} MB)")
                return True
        # Check current directory
        if os.path.exists("yolov8n-face.pt"):
            shutil.move("yolov8n-face.pt", dest)
            print(f"  ✅  yolov8n-face.pt saved ({os.path.getsize(dest)/1048576:.1f} MB)")
            return True
        print("  ✅  Model loaded by ultralytics (cached)")
        return True
    except ImportError:
        print("  ⚠️   ultralytics not installed: pip install ultralytics")
    except Exception as e:
        print(f"  ❌  ultralytics download failed: {e}")

    # Method 2: direct download from working mirror
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-face.pt",
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
    ]
    for url in urls:
        try:
            print(f"  ⬇️   Trying: {url}")
            urllib.request.urlretrieve(url, dest)
            if os.path.exists(dest) and os.path.getsize(dest) >= 5_000_000:
                print(f"  ✅  yolov8n-face.pt saved ({os.path.getsize(dest)/1048576:.1f} MB)")
                return True
        except Exception as e:
            print(f"  ❌  {str(e)[:60]}")
            if os.path.exists(dest):
                os.remove(dest)

    print("  ⚠️   Could not download yolov8n-face.pt")
    print("       EVA will use Haar Cascade as fallback (still works)")
    print("       Manual download: pip install ultralytics")
    print("                        python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download EVA model files")
    parser.add_argument("--check", action="store_true", help="Check status only")
    parser.add_argument("--force", action="store_true", help="Re-download all")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  EVA — Model Downloader")
    print("=" * 62)
    print(f"  Models directory: {MODELS_DIR}\n")

    if args.check:
        print_status()
        return

    for filename, meta in MODELS.items():
        if meta["url"] is None:
            continue   # handled separately below
        ok = download_model(filename, meta, force=args.force)
        if not ok and filename == "haarcascade_frontalface_default.xml":
            try_opencv_cascade()

    # YOLOv8-face uses special download
    download_yolov8_face()

    print()
    print_status()


if __name__ == "__main__":
    main()