"""
download_fer_dataset.py
──────────────────────
Download FER-2013 emotion dataset from alternative sources.

Usage:
  python download_fer_dataset.py              # auto-detect best source
  python download_fer_dataset.py --kaggle     # use Kaggle API
  python download_fer_dataset.py --direct     # direct download mirror
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "fer_data"
DOWNLOAD_DIR = Path(__file__).parent / "downloads"

def download_file(url, output_path, description=""):
    """Download file with progress bar."""
    print(f"⬇️  Downloading{' ' + description if description else ''}...")
    try:
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 // total_size, 100)
            bar_len = 40
            filled = int(bar_len * percent // 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {percent}% ({downloaded/1024/1024:.1f}MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, download_progress)
        print("\n  ✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract ZIP file."""
    print(f"📦 Extracting to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print("  ✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def download_kaggle():
    """Download FER-2013 using Kaggle API."""
    print("\n🔐 Using Kaggle API...")
    print("  Prerequisites: kaggle CLI + kaggle.json in ~/.kaggle/")
    try:
        os.system("kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset")
        zip_file = Path("face-expression-recognition-dataset.zip")
        if zip_file.exists():
            extract_zip(zip_file, DATASET_DIR)
            zip_file.unlink()  # delete after extraction
            return True
    except Exception as e:
        print(f"  ❌ Kaggle download failed: {e}")
    return False


def download_direct_mirror():
    """Download from alternative mirrors."""
    print("\n🌍 Trying direct download from mirror...")
    
    # Try multiple mirror sources
    mirrors = [
        ("https://data.mendeley.com/public-files/datasets/zft8gx74gr/files/1cad2a11-fa21-41c6-b236-fbb73b61dfe9/file_downloaded", "FER-2013 (Mendeley)"),
        ("https://www.dropbox.com/s/1ajwqvn2o2jrsqq/fer2013.tar.gz?dl=1", "FER-2013 (Dropbox)"),
    ]
    
    for url, desc in mirrors:
        print(f"\n  Trying: {desc}")
        output_file = DOWNLOAD_DIR / f"fer_dataset.zip"
        if download_file(url, output_file, desc):
            if extract_zip(output_file, DATASET_DIR):
                output_file.unlink()
                return True
    
    return False


def create_sample_dataset():
    """Create minimal sample FER dataset for testing."""
    print("\n🎨 Creating sample FER dataset structure...")
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    emotions = ["neutral", "happiness", "surprise", "sadness", 
                "anger", "disgust", "fear", "contempt"]
    
    for emotion in emotions:
        emotion_dir = DATASET_DIR / emotion
        emotion_dir.mkdir(exist_ok=True)
        
        # Create placeholder file
        placeholder = emotion_dir / "_placeholder.txt"
        placeholder.write_text(f"Place {emotion} face images here\n")
    
    print(f"  ✅ Dataset structure created at: {DATASET_DIR}")
    print("\n  📂 Structure:")
    for emotion in emotions:
        print(f"     fer_data/{emotion}/ (add .jpg files here)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download FER-2013 Emotion Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_fer_dataset.py              # auto-detect best method
  python download_fer_dataset.py --kaggle     # use Kaggle API (requires API key)
  python download_fer_dataset.py --direct     # try mirror sources
  python download_fer_dataset.py --sample     # create empty dataset structure
        """
    )
    parser.add_argument("--kaggle", action="store_true", help="Use Kaggle API")
    parser.add_argument("--direct", action="store_true", help="Try direct mirrors")
    parser.add_argument("--sample", action="store_true", help="Create sample structure only")
    args = parser.parse_args()
    
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    print("═" * 60)
    print("  FER-2013 Emotion Dataset Downloader")
    print("═" * 60)
    
    # Sample structure (always useful)
    if args.sample or (not args.kaggle and not args.direct):
        create_sample_dataset()
    
    # Try downloads if requested
    if args.kaggle:
        if not download_kaggle():
            download_direct_mirror()
    elif args.direct:
        download_direct_mirror()
    elif not args.sample:
        # Auto mode: try both
        print("\n🔄 Auto mode: trying best methods...")
        if not download_kaggle():
            download_direct_mirror()
    
    print("\n" + "═" * 60)
    print("  Evaluation Usage:")
    print("  python evaluation/evaluate_model.py --data ./fer_data")
    print("═" * 60)


if __name__ == "__main__":
    main()
