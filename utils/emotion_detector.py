"""
utils/emotion_detector.py  — v3.3 (AGGRESSIVE SMILE DETECTION)
────────────────────────────────────────────────────────────────
v3.3 IMPROVEMENTS (Aggressive Smile Detection):

  🎯 MASSIVE HAPPY BOOST: 3.50 (was 2.50)
     - 40% increase in happy calibration
     - Strongest boost of all emotions
  
  🔧 NEW: Surprise→Happy Correction
     - Smiles often misclassified as surprise
     - Transfers 40% of surprise to happy when both present
  
  🔧 NEW: Neutral→Happy Boost
     - Slight smiles read as neutral
     - Transfers 30% of neutral to happy when both present
  
  📉 REDUCED COMPETING EMOTIONS:
     - surprise: 1.00 → 0.90
     - neutral: 1.20 → 1.10
     - sad: 0.70 → 0.65
     - angry: 1.00 → 0.95
  
  📊 LOWER UNCERTAINTY: 0.18 → 0.15
     - Accepts more predictions instead of defaulting to neutral

v3.2 IMPROVEMENTS (Sad Bias Fix):

  🎯 SAD SUPPRESSION: Reduced calibration from 0.90 → 0.70
     - Fixes over-detection of sad emotion
     - Added sad→neutral/happy correction logic
  
  🎯 HAPPY BOOST: Increased from 2.20 → 2.50

v3.1 IMPROVEMENTS (Happy Detection Fix):

  🎯 HAPPY BOOST: Increased calibration from 1.80 → 2.20
     - More aggressive smile rescue (disgust → happy)
  
  🔄 BETTER SMOOTHING: 3-frame window (was 2)

v3 IMPROVEMENTS over v2:

  ✨ NEW: EVAEmotionClassifier (custom trained MobileNetV2)
     - 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
  
  ⚡ FASTER: YOLOv8n-face (5x faster)
  
  📊 SMOOTHER: 3-frame emotion smoothing
"""

import os, base64, logging
from collections import defaultdict, deque
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

# ── FER+ constants (legacy fallback model) ───────────────────
FERPLUS_EMOTIONS = ["neutral","happiness","surprise","sadness",
                    "anger","disgust","fear","contempt"]

EMOTION_MAP_FERPLUS = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
    "contempt":  "frustrated"
}

# ── EVA custom model constants ───────────────────────────────
# Class order MUST match training (alphabetical from ImageFolder)
EVA_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Map raw class names to dashboard-friendly names (already aligned)
EMOTION_MAP_EVA = {
    'angry':    'angry',
    'disgust':  'disgust',
    'fear':     'fear',
    'happy':    'happy',
    'neutral':  'neutral',
    'sad':      'sad',
    'surprise': 'surprise',
}

# ImageNet normalization (matches training preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

YAW_DISTRACTED_DEG  = 35
PITCH_DOWN_BORED_DEG = 28


# ══════════════════════════════════════════════════════════════
#  ✨ NEW: EVA CUSTOM EMOTION CLASSIFIER
# ══════════════════════════════════════════════════════════════

class EVAEmotionClassifier:
    """
    Custom-trained emotion classifier using YOUR FER-2013 model.
    
    Architecture: MobileNetV2 + custom head, 96x96 RGB input
    Trained: 28K images, 47.95% val accuracy on FER-2013
    
    Strengths: happy (96% conf), disgust, surprise, neutral
    Weaker on: fear (FER-2013 ground truth is noisy for fear)
    """
    
    INPUT_SIZE = 96
    
    def __init__(self):
        self._ready    = False
        self._session  = None
        self._inp_name = None
        self._out_name = None
        self._init()
    
    def _init(self):
        try:
            import onnxruntime as ort
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../models/eva_emotion_v1.onnx"
            )
            if not os.path.exists(model_path):
                logger.info("ℹ️  eva_emotion_v1.onnx not found — using FER+ only")
                return
            
            self._session  = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self._inp_name = self._session.get_inputs()[0].name
            self._out_name = self._session.get_outputs()[0].name
            self._ready    = True
            logger.info(f"✅ EVA custom emotion model loaded (val_acc=47.95%)")
            logger.info(f"   Classes: {EVA_CLASSES}")
        except Exception as e:
            logger.warning(f"⚠️  EVA model load failed: {e}")
    
    def predict(self, face_bgr) -> Dict:
        """
        Predict emotion from BGR face crop.
        Returns: {emotion, confidence, all_emotions, method}
        Returns None if not ready (caller should fall back).
        """
        if not self._ready:
            return None
        
        try:
            import cv2
            # ── Preprocess: BGR → RGB → resize → normalize → NCHW ──
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_AREA)
            arr = rgb.astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis, :]   # → (1, 3, 96, 96)
            arr = ((arr - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
            
            # ── Inference ──
            logits = self._session.run(None, {self._inp_name: arr})[0][0]
            
            # ── Temperature-scaled softmax (sharpens predictions) ──
            TEMPERATURE = 0.55                    # sharper
            exp = np.exp((logits - logits.max()) / TEMPERATURE)
            probs = exp / exp.sum()
            
            # ── Class-specific calibration (v7 — AGGRESSIVE happy boost) ──
            calibration = {
                'angry':    0.95,
                'disgust':  0.35,   # suppress (FER bug)
                'fear':     0.45,   # suppress (FER bug)
                'happy':    3.50,   # 🎯 MASSIVELY BOOSTED (was 2.50)
                'neutral':  1.10,   # reduced to favor happy
                'sad':      0.65,   # reduced more (was 0.70)
                'surprise': 0.90,   # reduced to favor happy
            }
            for i, cls in enumerate(EVA_CLASSES):
                probs[i] *= calibration[cls]
            probs = probs / probs.sum()
            
            # ── Smile rescue (disgust → happy when both present) ──
            happy_idx   = EVA_CLASSES.index('happy')
            disgust_idx = EVA_CLASSES.index('disgust')
            # 🎯 IMPROVED: More aggressive smile rescue
            if probs[disgust_idx] > 0.05 and probs[happy_idx] > 0.02:
                probs[happy_idx]  += probs[disgust_idx] * 0.95
                probs[disgust_idx] *= 0.05
            
            # ── Surprise → happy correction (smiles often misread as surprise) ──
            surprise_idx = EVA_CLASSES.index('surprise')
            # If surprise is high but happy exists, boost happy
            if probs[surprise_idx] > 0.20 and probs[happy_idx] > 0.05:
                transfer = probs[surprise_idx] * 0.40
                probs[happy_idx] += transfer
                probs[surprise_idx] -= transfer
            
            # ── Neutral → happy boost (slight smiles read as neutral) ──
            neutral_idx = EVA_CLASSES.index('neutral')
            # If neutral is high and happy exists, boost happy
            if probs[neutral_idx] > 0.20 and probs[happy_idx] > 0.08:
                transfer = probs[neutral_idx] * 0.30
                probs[happy_idx] += transfer
                probs[neutral_idx] -= transfer
            
            # ── Sad → neutral/happy correction (FER-2013 sad bias) ──
            sad_idx = EVA_CLASSES.index('sad')
            # If sad is detected but neutral or happy are also present, reduce sad
            if probs[sad_idx] > 0.15:
                if probs[neutral_idx] > 0.10 or probs[happy_idx] > 0.08:
                    # Redistribute sad probability to neutral/happy
                    transfer = probs[sad_idx] * 0.35
                    if probs[happy_idx] > probs[neutral_idx]:
                        probs[happy_idx] += transfer * 0.8
                        probs[neutral_idx] += transfer * 0.2
                    else:
                        probs[neutral_idx] += transfer * 0.6
                        probs[happy_idx] += transfer * 0.4
                    probs[sad_idx] *= 0.65
            
            # ── Fear → surprise (only when surprise also strong) ──
            fear_idx     = EVA_CLASSES.index('fear')
            # 🎯 IMPROVED: Don't steal from happy
            if probs[fear_idx] > 0.20 and probs[surprise_idx] > 0.20 and probs[happy_idx] < 0.12:
                probs[surprise_idx] += probs[fear_idx] * 0.40
                probs[fear_idx]     *= 0.60
            
            probs = probs / probs.sum()
            
            # ── Lighter margin check (🎯 VERY LOW threshold) ──────
            if probs.max() < 0.15:  # was 0.18, now even lower
                all_emo = {EVA_CLASSES[i]: float(probs[i]) for i in range(len(EVA_CLASSES))}
                return {
                    "emotion":      "neutral",
                    "confidence":   0.50,
                    "all_emotions": all_emo,
                    "method":       "eva_v1_uncertain"
                }
            
            # ── Pick top class ──
            top_idx = int(np.argmax(probs))
            raw     = EVA_CLASSES[top_idx]
            emotion = EMOTION_MAP_EVA.get(raw, raw)
            confidence = float(probs[top_idx])
            
            all_emotions = {
                EMOTION_MAP_EVA.get(EVA_CLASSES[i], EVA_CLASSES[i]): float(probs[i])
                for i in range(len(EVA_CLASSES))
            }
            
            top3 = sorted(zip(EVA_CLASSES, probs.tolist()),
                          key=lambda x: -x[1])[:3]
            logger.info(
                f"🎭 EVA model: {raw} → {emotion} ({confidence*100:.1f}%) | "
                f"top3: {[(e, f'{p*100:.1f}%') for e, p in top3]}"
            )
            
            return {
                "emotion":      emotion,
                "confidence":   round(confidence, 3),
                "all_emotions": all_emotions,
                "method":       "eva_v1"
            }
        except Exception as e:
            logger.error(f"EVA model inference error: {e}")
            return None
    
    def is_ready(self) -> bool:
        return self._ready


# ══════════════════════════════════════════════════════════════
#  IMPROVEMENT 1 — FACE PREPROCESSOR (unchanged from v2)
# ══════════════════════════════════════════════════════════════

class FacePreprocessor:
    """Better preprocessing for FER+ fallback model (grayscale 64x64)."""
    
    def __init__(self):
        self._lm_ready = False
        self._init_dlib()

    def _init_dlib(self):
        try:
            import dlib
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../models/shape_predictor_68_face_landmarks.dat"
            )
            if os.path.exists(model_path):
                self._lm_detector = dlib.shape_predictor(model_path)
                self._lm_ready    = True
                logger.info("✅ dlib face alignment ready")
            else:
                logger.info("ℹ️  shape_predictor_68.dat not found — face alignment skipped")
        except ImportError:
            logger.info("ℹ️  dlib not installed — using OpenCV preprocessing only")

    def preprocess(self, face_bgr, target_size: int = 64) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) \
               if len(face_bgr.shape) == 3 else face_bgr.copy()
        gray = self._gamma_correct(gray, gamma=1.3)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=55, sigmaSpace=55)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        if self._lm_ready:
            aligned = self._align(face_bgr, gray.copy(), target_size)
            if aligned is not None:
                return aligned

        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        return gray.astype(np.float32)

    def _gamma_correct(self, gray: np.ndarray, gamma: float = 1.3) -> np.ndarray:
        import cv2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(gray, table)

    def _align(self, face_bgr, gray_processed, target_size):
        try:
            import cv2, dlib
            h, w = gray_processed.shape
            rect = dlib.rectangle(0, 0, w, h)
            gray_orig = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray_rsz  = cv2.resize(gray_orig, (w, h))
            lm = self._lm_detector(gray_rsz, rect)

            lx = np.mean([(lm.part(i).x) for i in range(36, 42)])
            ly = np.mean([(lm.part(i).y) for i in range(36, 42)])
            rx = np.mean([(lm.part(i).x) for i in range(42, 48)])
            ry = np.mean([(lm.part(i).y) for i in range(42, 48)])

            angle      = np.degrees(np.arctan2(ry - ly, rx - lx))
            eye_center = ((lx + rx) / 2, (ly + ry) / 2)

            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            rotated = cv2.warpAffine(gray_processed, M, (w, h))
            resized = cv2.resize(rotated, (target_size, target_size))
            return cv2.equalizeHist(resized).astype(np.float32)
        except Exception as e:
            logger.debug(f"Alignment skipped: {e}")
            return None

    def alignment_ready(self) -> bool:
        return self._lm_ready


# ══════════════════════════════════════════════════════════════
#  IMPROVEMENT 2 — HEAD POSE ESTIMATOR (unchanged from v2)
# ══════════════════════════════════════════════════════════════

class HeadPoseEstimator:
    """MediaPipe FaceMesh-based head orientation estimator."""

    def __init__(self):
        self._ready       = False
        self._face_mesh   = None
        self._api_version = "none"
        self._init()

    def _init(self):
        if self._init_new_api():
            return
        if self._init_legacy_api():
            return
        logger.info("ℹ️  mediapipe not available — head pose disabled")

    def _init_new_api(self) -> bool:
        try:
            import mediapipe as mp
            if not hasattr(mp, 'tasks'):
                return False
            import urllib.request
            model_path = os.path.join(
                os.path.dirname(__file__),
                "../models/face_landmarker.tflite"
            )
            if not os.path.exists(model_path):
                url = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                logger.info("⬇️  Downloading MediaPipe face landmarker model (~3MB)...")
                try:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    urllib.request.urlretrieve(url, model_path)
                    logger.info("✅ face_landmarker.tflite downloaded")
                except Exception as e:
                    logger.warning(f"⚠️  Could not download face landmarker: {e}")
                    return False
            if not os.path.exists(model_path):
                return False

            BaseOptions   = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            self._face_mesh   = FaceLandmarker.create_from_options(options)
            self._api_version = "new"
            self._ready       = True
            logger.info("✅ MediaPipe head pose ready (Tasks API 0.10.x)")
            return True
        except Exception as e:
            logger.debug(f"New MediaPipe API init failed: {e}")
            return False

    def _init_legacy_api(self) -> bool:
        try:
            import mediapipe as mp
            if not hasattr(mp, 'solutions'):
                return False
            mp_fm = mp.solutions.face_mesh
            self._face_mesh = mp_fm.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5
            )
            self._api_version = "legacy"
            self._ready       = True
            logger.info("✅ MediaPipe head pose ready (legacy solutions API)")
            return True
        except Exception as e:
            logger.debug(f"Legacy MediaPipe API init failed: {e}")
            return False

    def estimate(self, face_bgr) -> Dict:
        if not self._ready or self._face_mesh is None:
            return {"pose_available": False}
        try:
            import cv2
            rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            lm = self._get_landmarks(rgb, h, w)
            if lm is None:
                return {"pose_available": False, "reason": "no_landmarks"}

            face_3d = np.array([
                [ 0.0,     0.0,    0.0],
                [ 0.0,  -330.0,  -65.0],
                [-225.0,  170.0, -135.0],
                [ 225.0,  170.0, -135.0],
                [-150.0, -150.0, -125.0],
                [ 150.0, -150.0, -125.0],
            ], dtype=np.float64)
            idx2d  = [1, 152, 263, 33, 287, 57]
            face_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in idx2d],
                               dtype=np.float64)

            focal   = w
            cam_mat = np.array([[focal, 0, w/2],[0, focal, h/2],[0, 0, 1]],
                               dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(face_3d, face_2d, cam_mat,
                                        np.zeros((4,1), dtype=np.float64))
            if not ok:
                return {"pose_available": False}

            rmat, _    = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(rmat)
            pitch = angles[0] * 360
            yaw   = angles[1] * 360
            roll  = angles[2] * 360
            attention = self._classify(yaw, pitch)
            return {
                "pose_available": True,
                "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1),
                "attention_state": attention
            }
        except Exception as e:
            logger.debug(f"Head pose error: {e}")
            return {"pose_available": False}

    def _get_landmarks(self, rgb_img, h, w):
        try:
            if self._api_version == "new":
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                result = self._face_mesh.detect(mp_image)
                if not result.face_landmarks:
                    return None
                return result.face_landmarks[0]
            else:
                result = self._face_mesh.process(rgb_img)
                if not result.multi_face_landmarks:
                    return None
                return result.multi_face_landmarks[0].landmark
        except Exception as e:
            logger.debug(f"Landmark detection error: {e}")
            return None

    def _classify(self, yaw: float, pitch: float) -> str:
        if abs(yaw) > YAW_DISTRACTED_DEG:    return "looking_away"
        if pitch < -PITCH_DOWN_BORED_DEG:    return "looking_down"
        if pitch > 15:                       return "looking_up"
        return "attentive"

    def is_ready(self) -> bool:
        return self._ready


# ══════════════════════════════════════════════════════════════
#  ⚡ YOLO FACE DETECTOR (now uses yolov8n-face by default)
# ══════════════════════════════════════════════════════════════

class YOLOFaceDetector:
    """
    YOLO face detector — prefers yolov8n-face.pt (5x faster than yolov8m).
    Falls back to yolov8m if face-specific model not available.
    """

    def __init__(self):
        self._ready      = False
        self._model      = None
        self._mode       = None
        self._init()

    def _init(self):
        # ── Prefer face-specific ONNX (fastest) ────────────────
        onnx_path = os.path.join(
            os.path.dirname(__file__),
            "../models/yolov8n-face.onnx"
        )
        if os.path.exists(onnx_path) and self._init_onnx(onnx_path):
            return

        # ── Try ultralytics with yolov8n-face.pt (preferred) ───
        try:
            from ultralytics import YOLO
            face_pt_paths = [
                os.path.join(os.path.dirname(__file__), "../yolov8n-face.pt"),
                os.path.join(os.path.dirname(__file__), "../models/yolov8n-face.pt"),
                "yolov8n-face.pt",
            ]
            for p in face_pt_paths:
                if os.path.exists(p):
                    logger.info(f"⬇️  Loading yolov8n-face.pt from {p}...")
                    self._model = YOLO(p)
                    self._mode  = "ultralytics_face"
                    self._ready = True
                    logger.info("✅ YOLOv8n-FACE detector ready (5x faster)")
                    return

            # ── Fallback to general yolov8m ─────────────────────
            logger.info("⬇️  yolov8n-face.pt not found, loading yolov8m.pt...")
            self._model = YOLO("yolov8m.pt")
            self._mode  = "ultralytics_general"
            self._ready = True
            logger.info("✅ YOLOv8m general detector ready (slower)")

        except ImportError:
            logger.info("ℹ️  ultralytics not installed — Haar Cascade fallback only")
        except Exception as e:
            logger.warning(f"⚠️  YOLOv8 init error: {e}")

    def _init_onnx(self, model_path: str) -> bool:
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(model_path)
            self._mode        = "onnx"
            self._ready       = True
            logger.info("✅ YOLOv8n-face ready (ONNX mode)")
            return True
        except Exception as e:
            logger.debug(f"ONNX face detector init failed: {e}")
            return False

    def detect(self, frame_bgr) -> list:
        if not self._ready:
            return []
        try:
            if self._mode in ("ultralytics_face", "ultralytics_general"):
                return self._detect_ultralytics(frame_bgr)
            elif self._mode == "onnx":
                return self._detect_onnx(frame_bgr)
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
        return []

    def _detect_ultralytics(self, frame_bgr) -> list:
        # face model: only 1 class (face). general model: filter to "person" (class 0)
        results = self._model(
            frame_bgr,
            conf=0.40 if self._mode == "ultralytics_face" else 0.50,
            iou=0.40,
            verbose=False,
            imgsz=640,
            classes=None if self._mode == "ultralytics_face" else [0],
        )
        faces = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                if conf >= 0.35:
                    x, y = max(0, x1), max(0, y1)
                    w, h = x2 - x1, y2 - y1
                    if w > 20 and h > 20:
                        faces.append((x, y, w, h))
        faces.sort(key=lambda f: f[2]*f[3], reverse=True)
        return faces

    def _detect_onnx(self, frame_bgr) -> list:
        import cv2
        h_orig, w_orig = frame_bgr.shape[:2]
        size = 320
        img   = cv2.resize(frame_bgr, (size, size))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp   = img.astype(np.float32) / 255.0
        inp   = inp.transpose(2, 0, 1)[np.newaxis]
        inp_name = self._ort_session.get_inputs()[0].name
        outputs  = self._ort_session.run(None, {inp_name: inp})
        preds    = outputs[0][0].T

        faces = []
        for pred in preds:
            cx, cy, pw, ph, conf = pred[:5]
            if conf < 0.35:
                continue
            x1 = int((cx - pw/2) / size * w_orig)
            y1 = int((cy - ph/2) / size * h_orig)
            w  = int(pw / size * w_orig)
            h  = int(ph / size * h_orig)
            if w > 20 and h > 20:
                faces.append((max(0,x1), max(0,y1), w, h))
        faces.sort(key=lambda f: f[2]*f[3], reverse=True)
        return faces

    def is_ready(self) -> bool:
        return self._ready

    def model_type(self) -> str:
        return self._mode or "none"


# ══════════════════════════════════════════════════════════════
#  📊 SMOOTHING BUFFER (reduces dashboard flicker)
# ══════════════════════════════════════════════════════════════

class EmotionSmoother:
    """
    Per-face rolling buffer of last N predictions.
    Returns the most-frequent emotion to reduce flicker on dashboard.
    """
    WINDOW = 3  # 🎯 INCREASED from 2 to 3 for better stability
    def __init__(self):
        self._buffers = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def smooth(self, key: str, emotion: str, confidence: float) -> tuple:
        """Returns (smoothed_emotion, smoothed_confidence)."""
        self._buffers[key].append((emotion, confidence))
        buf = list(self._buffers[key])
        if len(buf) < 2:
            return emotion, confidence

        # Vote on most-frequent emotion in window (weighted by confidence)
        counts = defaultdict(float)
        for emo, conf in buf:
            counts[emo] += conf  # weight by confidence
        winner = max(counts.items(), key=lambda x: x[1])[0]
        avg_conf = np.mean([c for e, c in buf if e == winner])
        
        logger.debug(f"🔄 Smoothing [{key}]: raw={emotion}, smoothed={winner}, buffer={[e for e,c in buf]}")
        return winner, float(avg_conf)


# ══════════════════════════════════════════════════════════════
#  MAIN EMOTION DETECTOR
# ══════════════════════════════════════════════════════════════

class EmotionDetector:
    """
    v3: EVA custom model (primary) + FER+ (fallback) + head pose + smoothing.
    """

    def __init__(self):
        self._onnx_ready   = False
        self._cv_ready     = False
        self._ocr_ready    = False
        self._session      = None
        self._face_cascade = None
        self._preprocessor    = FacePreprocessor()
        self._pose_estimator  = HeadPoseEstimator()
        self._yolo_detector   = YOLOFaceDetector()
        self._eva_classifier  = EVAEmotionClassifier()   # ✨ NEW
        self._smoother        = EmotionSmoother()         # 📊 NEW
        self._init_ferplus_onnx()
        self._init_cv()
        self._init_ocr()

    # ── Init ──────────────────────────────────────────────────

    def _init_ferplus_onnx(self):
        try:
            import onnxruntime as ort
            model_path = os.path.join(
                os.path.dirname(__file__), "../models/emotion_ferplus_8.onnx"
            )
            if os.path.exists(model_path):
                self._session    = ort.InferenceSession(model_path)
                self._onnx_ready = True
                logger.info("✅ FER+ ONNX model loaded (fallback)")
            else:
                logger.warning("⚠️  FER+ ONNX model not found")
        except ImportError:
            logger.warning("⚠️  onnxruntime not installed")

    def _init_cv(self):
        try:
            import cv2
            cascade_path = os.path.join(
                os.path.dirname(__file__),
                "../models/haarcascade_frontalface_default.xml"
            )
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            self._cv_ready     = True
            logger.info("✅ OpenCV face cascade loaded")
        except ImportError:
            logger.warning("⚠️  opencv not installed")

    def _init_ocr(self):
        try:
            import pytesseract  # noqa
            self._ocr_ready = True
            logger.info("✅ OCR (pytesseract) ready")
        except ImportError:
            logger.warning("⚠️  pytesseract not installed")

    # ── Public API ────────────────────────────────────────────

    def detect_from_base64(self, image_b64: str, smooth_key: str = None) -> Dict:
        """
        Main entry point.
        Optional smooth_key (e.g. student name) enables 3-frame smoothing.
        """
        if not self._cv_ready:
            return {"emotion": "neutral", "confidence": 0.5, "method": "unavailable"}
        try:
            import cv2
            frame = self._decode_image(image_b64)
            if frame is None:
                return {"emotion": "neutral", "confidence": 0.5}

            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray  = clahe.apply(gray)

            faces = self._detect_faces(gray, frame_bgr=frame)
            if len(faces) == 0:
                return {"emotion": "neutral", "confidence": 0.3, "face_found": False}

            x, y, w, h = faces[0]
            h_f, w_f   = frame.shape[:2]
            x, y       = max(0, x), max(0, y)
            w, h       = min(w, w_f-x), min(h, h_f-y)
            face_crop  = frame[y:y+h, x:x+w]

            emotion_r = self._classify(face_crop)
            pose_r    = self._pose_estimator.estimate(face_crop)
            result    = self._fuse(emotion_r, pose_r)

            # ── Apply smoothing if key provided ─────────────────
            if smooth_key:
                smooth_emo, smooth_conf = self._smoother.smooth(
                    smooth_key, result["emotion"], result["confidence"]
                )
                result["emotion_raw"]   = result["emotion"]
                result["confidence_raw"] = result["confidence"]
                result["emotion"]    = smooth_emo
                result["confidence"] = round(smooth_conf, 3)

            result["face_found"] = True
            result["bbox"]       = [int(x), int(y), int(w), int(h)]
            return result

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"emotion": "neutral", "confidence": 0.5}

    # ── Internal ──────────────────────────────────────────────

    def _detect_faces(self, gray, frame_bgr=None):
        """Face detection: YOLO primary → Haar fallback."""
        if self._yolo_detector.is_ready() and frame_bgr is not None:
            faces = self._yolo_detector.detect(frame_bgr)
            if faces:
                return faces

        if self._cv_ready and self._face_cascade is not None:
            for sf, mn, ms in [(1.1,4,(30,30)),(1.05,3,(20,20)),(1.03,2,(15,15))]:
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn, minSize=ms
                )
                if len(faces):
                    return [tuple(f) for f in faces]
        return []

    def _classify(self, face_bgr) -> Dict:
        """
        ✨ Try EVA custom model first, fall back to FER+ on failure.
        """
        # ── PRIMARY: EVA custom model ─────────────────────────
        if self._eva_classifier.is_ready():
            result = self._eva_classifier.predict(face_bgr)
            if result is not None:
                return result

        # ── FALLBACK: FER+ ONNX ────────────────────────────────
        if self._onnx_ready and self._session:
            return self._classify_ferplus(face_bgr)

        # ── LAST RESORT: brightness heuristic ──────────────────
        return self._fallback(face_bgr)

    def _classify_ferplus(self, face_bgr) -> Dict:
        """Original FER+ classifier (kept as fallback)."""
        try:
            processed = self._preprocessor.preprocess(face_bgr, 64)
            inp_name  = self._session.get_inputs()[0].name
            inp       = processed.reshape(1, 1, 64, 64)
            scores    = self._session.run(None, {inp_name: inp})[0][0]

            exp   = np.exp(scores - np.max(scores))
            probs = exp / exp.sum()

            top_idx     = int(np.argmax(probs))
            emotion_raw = FERPLUS_EMOTIONS[top_idx]
            emotion     = EMOTION_MAP_FERPLUS.get(emotion_raw, "neutral")
            confidence  = float(probs[top_idx])

            all_emotions = {
                EMOTION_MAP_FERPLUS.get(FERPLUS_EMOTIONS[i], FERPLUS_EMOTIONS[i]): float(probs[i])
                for i in range(len(FERPLUS_EMOTIONS))
            }
            logger.info(
                f"FER+ fallback: {emotion_raw} → {emotion} ({confidence*100:.1f}%)"
            )
            return {
                "emotion":      emotion,
                "confidence":   round(confidence, 3),
                "all_emotions": all_emotions,
                "method":       "ferplus_v2"
            }
        except Exception as e:
            logger.error(f"FER+ ONNX error: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "method": "error"}

    def _fuse(self, emotion_r: Dict, pose_r: Dict) -> Dict:
        """Fusion: emotion is primary, pose is informational."""
        emotion    = emotion_r.get("emotion", "neutral")
        confidence = emotion_r.get("confidence", 0.5)

        if not pose_r.get("pose_available", False):
            return {**emotion_r, "pose": None, "attention": "unknown"}

        attention = pose_r.get("attention_state", "attentive")
        if attention == "attentive":
            confidence = min(0.99, confidence * 1.02)

        return {
            **emotion_r,
            "emotion":    emotion,
            "confidence": round(confidence, 3),
            "pose": {
                "yaw":       pose_r.get("yaw"),
                "pitch":     pose_r.get("pitch"),
                "roll":      pose_r.get("roll"),
                "attention": attention
            },
            "pose_override": False,
            "attention":     attention
        }

    def _fallback(self, face_img) -> Dict:
        try:
            import cv2
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            b    = np.mean(gray)
            if b > 160: return {"emotion":"happy",   "confidence":0.55,"method":"heuristic"}
            if b <  80: return {"emotion":"sad",     "confidence":0.50,"method":"heuristic"}
            return             {"emotion":"neutral", "confidence":0.60,"method":"heuristic"}
        except Exception:
            return             {"emotion":"neutral", "confidence":0.50,"method":"fallback"}

    def _decode_image(self, image_b64: str):
        import cv2
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        arr = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _read_name_label(self, screen, x, y, w, h) -> str:
        if not self._ocr_ready:
            return ""
        try:
            import cv2, pytesseract
            lr = screen[y+h: y+h + max(20,h//5), x:x+w]
            if lr.size == 0:
                return ""
            gray = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
            _, t = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            txt  = pytesseract.image_to_string(t, config="--psm 7 --oem 3").strip()
            return "".join(c for c in txt if c.isalpha() or c in " -.").strip()[:30]
        except Exception:
            return ""

    def detect_all_faces_from_screen(self, screen_b64: str) -> List[Dict]:
        if not self._cv_ready:
            return []
        try:
            import cv2
            screen = self._decode_image(screen_b64)
            if screen is None:
                return []
            gray  = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(40,40)
            )
            results = []
            for i, (x,y,w,h) in enumerate(faces):
                crop  = screen[y:y+h, x:x+w]
                er    = self._classify(crop)
                pr    = self._pose_estimator.estimate(crop)
                fused = self._fuse(er, pr)
                name  = self._read_name_label(screen, x, y, w, h)
                results.append({"tile_index":i,"bbox":[int(x),int(y),int(w),int(h)],
                                 "name_ocr":name or f"Student_{i+1}",**fused})
            return results
        except Exception as e:
            logger.error(f"Multi-face error: {e}")
            return []

    def is_ready(self) -> dict:
        return {
            "eva_custom_model": self._eva_classifier.is_ready(),
            "ferplus_fallback": self._onnx_ready,
            "opencv":           self._cv_ready,
            "ocr":              self._ocr_ready,
            "face_alignment":   self._preprocessor.alignment_ready(),
            "head_pose":        self._pose_estimator.is_ready(),
            "yolo_face":        self._yolo_detector.is_ready(),
            "yolo_type":        self._yolo_detector.model_type(),
            "primary_model":    "eva_v1" if self._eva_classifier.is_ready() else "ferplus"
        }