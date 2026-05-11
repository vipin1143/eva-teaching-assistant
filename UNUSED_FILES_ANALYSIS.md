# 📋 EVA Project - Unused Files Analysis

## Summary

This document identifies files in the EVA project that are **NOT used during runtime** but serve other purposes (training, evaluation, testing, documentation).

---

## ✅ **RUNTIME FILES** (Used by app.py)

These files are actively used when running the application:

### Core Application
- ✅ `app.py` - Main entry point
- ✅ `.env` - Environment configuration

### Backend
- ✅ `backend/server.py` - FastAPI server with all routes
- ✅ `backend/__init__.py` - Backend package init
- ❌ `backend/zoom_setup.py` - **UNUSED** (manual setup script, not imported)

### Database
- ✅ `database/db_manager.py` - Database operations
- ✅ `database/__init__.py` - Database package init
- ✅ `database/eva.db` - SQLite database file
- ✅ `database/eva.db-shm` - SQLite shared memory
- ✅ `database/eva.db-wal` - SQLite write-ahead log

### Utils (All actively used)
- ✅ `utils/emotion_detector.py` - Emotion detection (v3.1)
- ✅ `utils/text_sentiment.py` - Chat sentiment analysis
- ✅ `utils/student_tracker.py` - Student state tracking
- ✅ `utils/notification_engine.py` - Alert system
- ✅ `utils/screen_capture.py` - Screen capture for Zoom/Meet
- ✅ `utils/ai_suggestion.py` - AI teaching suggestions
- ✅ `utils/session_report.py` - Session report generation
- ✅ `utils/__init__.py` - Utils package init
- ❌ `utils/ai_tutor.py` - **UNUSED** (not imported anywhere)
- ❌ `utils/preprocessing.py` - **UNUSED** (only used by evaluation scripts)

### Frontend
- ✅ `teacher_dashboard/index.html` - Teacher UI
- ✅ `student_link/index.html` - Student UI

### Models (Runtime)
- ✅ `models/eva_emotion_v1.onnx` - Custom emotion model (primary)
- ✅ `models/emotion_ferplus_8.onnx` - FER+ fallback model
- ✅ `models/haarcascade_frontalface_default.xml` - Face detection fallback
- ✅ `models/face_landmarker.tflite` - MediaPipe head pose
- ✅ `models/shape_predictor_68_face_landmarks.dat` - dlib face alignment
- ✅ `yolov8n-face.pt` - YOLO face detector (primary)

---

## ❌ **UNUSED FILES** (Not Used During Runtime)

### 1. Training Scripts
These are used ONLY for training new models, not during runtime:

- ❌ `train_emotion_model.py` - Train custom emotion model
- ❌ `models/train_emotion_model.py` - Alternative training script
- ❌ `download_fer_dataset.py` - Download FER-2013 dataset
- ❌ `verify_dataset.py` - Verify dataset structure

**Purpose:** Model training (one-time or periodic retraining)
**Can be deleted?** ⚠️ Keep if you plan to retrain models, delete if using pre-trained only

---

### 2. Evaluation Scripts
These are used ONLY for testing model accuracy:

- ❌ `evaluate_model_improved.py` - Evaluate with advanced preprocessing
- ❌ `evaluation/evaluate_model.py` - Basic model evaluation
- ❌ `evaluation/evaluate_yolo_classifier.py` - YOLO classifier evaluation
- ❌ `evaluation/__init__.py` - Evaluation package init

**Purpose:** Model performance testing
**Can be deleted?** ⚠️ Keep for quality assurance, delete if not needed

---

### 3. Export/Conversion Scripts
These are used ONLY for converting models:

- ❌ `export_to_onnx.py` - Convert PyTorch model to ONNX

**Purpose:** Model format conversion (one-time)
**Can be deleted?** ⚠️ Keep if you might retrain, delete otherwise

---

### 4. Model Download Scripts
These are used ONLY for downloading model files:

- ❌ `models/download_models.py` - Download required models
- ❌ `models/download_yolo.py` - Download YOLO models
- ❌ `models/__init__.py` - Models package init (empty)
- ❌ `models/models.py` - Model definitions (if exists)

**Purpose:** Initial setup (one-time)
**Can be deleted?** ⚠️ Keep for easy setup on new machines, delete if all models present

---

### 5. Test Scripts
These are used ONLY for testing:

- ❌ `test_happy_detection.py` - Test happy emotion detection
- ❌ `test_onnx_model.py` - Test ONNX model loading

**Purpose:** Development testing
**Can be deleted?** ✅ Yes, safe to delete after testing

---

### 6. Documentation Files
These are reference documents:

- ❌ `README.md` - Project documentation
- ❌ `HAPPY_DETECTION_FIX.md` - Happy detection fix documentation
- ❌ `TRAIN_EMOTION_MODEL_GUIDE.md` - Training guide
- ❌ `YOLO_ACCURACY_GUIDE.md` - YOLO accuracy guide
- ❌ `accuracy_report.txt` - Accuracy report
- ❌ `confusion_matrix.png` - Confusion matrix visualization

**Purpose:** Documentation and reference
**Can be deleted?** ⚠️ Keep for reference, delete if not needed

---

### 7. Unused Utility Files
These are NOT imported by any runtime code:

- ❌ `utils/ai_tutor.py` - AI tutor (not imported anywhere)
- ❌ `utils/preprocessing.py` - Advanced preprocessing (only used by evaluation)

**Purpose:** Planned features or evaluation only
**Can be deleted?** ✅ Yes, safe to delete (or keep for future use)

---

### 8. Training Artifacts
These are generated during training:

- ❌ `models/eva_emotion_v1_best.pt` - PyTorch model weights (source)
- ❌ `models/eva_emotion_v1_meta.json` - Model metadata
- ❌ `models/eva_emotion_v1.onnx.data` - ONNX data file
- ❌ `runs/emotion-cls/` - Training run logs

**Purpose:** Training artifacts (ONNX is used, .pt is source)
**Can be deleted?** ⚠️ Keep .pt if you might retrain, delete runs/ folder

---

### 9. Dataset Folders
These contain training/validation images:

- ❌ `images/train/` - Training dataset (28K+ images)
- ❌ `images/validation/` - Validation dataset
- ❌ `images/images/` - Duplicate dataset structure

**Purpose:** Model training data
**Can be deleted?** ✅ Yes, safe to delete if not retraining (saves ~2GB)

---

### 10. Setup/Config Files
- ❌ `requirement.txt` - Python dependencies (typo: should be requirements.txt)
- ❌ `backend/zoom_setup.py` - Zoom webhook setup script

**Purpose:** Initial setup
**Can be deleted?** ⚠️ Keep requirement.txt, zoom_setup.py can be deleted

---

## 🎯 **SAFE TO DELETE** (Won't affect runtime)

If you want to clean up the project and only keep runtime files:

```bash
# Test scripts
rm eva-project/test_happy_detection.py
rm eva-project/test_onnx_model.py

# Training scripts (if not retraining)
rm eva-project/train_emotion_model.py
rm eva-project/download_fer_dataset.py
rm eva-project/verify_dataset.py
rm eva-project/export_to_onnx.py

# Evaluation scripts (if not testing)
rm -rf eva-project/evaluation/

# Model download scripts (if models already present)
rm eva-project/models/download_models.py
rm eva-project/models/download_yolo.py
rm eva-project/models/train_emotion_model.py

# Unused utils
rm eva-project/utils/ai_tutor.py
rm eva-project/utils/preprocessing.py

# Training artifacts (keep ONNX, delete PyTorch source)
rm eva-project/models/eva_emotion_v1_best.pt
rm eva-project/models/eva_emotion_v1_meta.json
rm eva-project/models/eva_emotion_v1.onnx.data
rm -rf eva-project/runs/

# Dataset (if not retraining - saves ~2GB)
rm -rf eva-project/images/

# Documentation (if not needed)
rm eva-project/HAPPY_DETECTION_FIX.md
rm eva-project/TRAIN_EMOTION_MODEL_GUIDE.md
rm eva-project/YOLO_ACCURACY_GUIDE.md
rm eva-project/accuracy_report.txt
rm eva-project/confusion_matrix.png

# Setup scripts
rm eva-project/backend/zoom_setup.py
```

---

## ⚠️ **KEEP THESE** (Required for runtime)

**Minimum files needed to run EVA:**

```
eva-project/
├── app.py                          ✅ Main entry
├── .env                            ✅ Config
├── backend/
│   ├── server.py                   ✅ API server
│   └── __init__.py                 ✅ Package
├── database/
│   ├── db_manager.py               ✅ Database
│   ├── __init__.py                 ✅ Package
│   └── eva.db                      ✅ Data
├── utils/
│   ├── emotion_detector.py         ✅ Emotion detection
│   ├── text_sentiment.py           ✅ Sentiment
│   ├── student_tracker.py          ✅ Tracking
│   ├── notification_engine.py      ✅ Alerts
│   ├── screen_capture.py           ✅ Screen capture
│   ├── ai_suggestion.py            ✅ AI suggestions
│   ├── session_report.py           ✅ Reports
│   └── __init__.py                 ✅ Package
├── models/
│   ├── eva_emotion_v1.onnx         ✅ Primary model
│   ├── emotion_ferplus_8.onnx      ✅ Fallback model
│   ├── haarcascade_frontalface_default.xml  ✅ Face detection
│   ├── face_landmarker.tflite      ✅ Head pose
│   └── shape_predictor_68_face_landmarks.dat ✅ Alignment
├── yolov8n-face.pt                 ✅ YOLO face detector
├── teacher_dashboard/
│   └── index.html                  ✅ Teacher UI
├── student_link/
│   └── index.html                  ✅ Student UI
└── README.md                       ⚠️ Optional (documentation)
```

---

## 📊 **Storage Savings**

If you delete all unused files:

- **Training dataset**: ~2 GB
- **Training artifacts**: ~50 MB
- **Evaluation scripts**: ~100 KB
- **Test scripts**: ~50 KB
- **Documentation**: ~500 KB

**Total savings: ~2.05 GB**

---

## 🔍 **How to Verify**

To verify a file is unused, search for imports:

```bash
# Check if a file is imported anywhere
grep -r "from.*filename" eva-project/
grep -r "import.*filename" eva-project/

# Example: Check if ai_tutor is used
grep -r "ai_tutor" eva-project/
# Result: Only found in ai_tutor.py itself = UNUSED
```

---

## ✅ **Recommendation**

**For production deployment:**
- Delete: Test scripts, training scripts, evaluation scripts, dataset
- Keep: All runtime files, README.md, requirement.txt

**For development:**
- Keep everything for flexibility

**For minimal deployment:**
- Use the "KEEP THESE" list above
- Saves ~2GB of space
