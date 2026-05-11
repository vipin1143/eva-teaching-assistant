"""
Student Tracker
Maintains real-time emotion state for every student in the session.
Handles multi-source fusion (face, screen capture, chat).
"""
from datetime import datetime
from typing import Dict, List, Optional

EMOTION_TO_STATE = {
    "happy":    {"state": "engaged",     "alert_level": 0},
    "surprise": {"state": "curious",     "alert_level": 0},
    "neutral":  {"state": "neutral",     "alert_level": 0},
    "sad":      {"state": "disengaged",  "alert_level": 1},
    "fear":     {"state": "anxious",     "alert_level": 2},
    "angry":    {"state": "frustrated",  "alert_level": 2},
    "disgust":  {"state": "averse",      "alert_level": 1},
    "confused": {"state": "confused",    "alert_level": 2},
    "bored":    {"state": "disengaged",  "alert_level": 1},
    "engaged":  {"state": "engaged",     "alert_level": 0},
}

ALERT_EMOJI = {0: "🟢", 1: "🟡", 2: "🔴"}
EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
    "surprise": "😲", "disgust": "🤢", "neutral": "😐",
    "confused": "🤔", "bored": "😒", "engaged": "🙂",
    "frustrated": "😤", "anxious": "😰", "curious": "🔍",
    "disengaged": "😶"
}


class StudentState:
    def __init__(self, name: str):
        self.name = name
        self.emotion = "neutral"
        self.confidence = 0.5
        self.learning_state = "neutral"
        self.alert_level = 0
        self.emotion_history: List[str] = []
        self.history_size = 5
        self.current_emotion_start = datetime.now()
        self.duration_seconds = 0
        self.last_message = ""
        self.last_updated = datetime.now()
        self.source = "none"
        self.notification_cooldown: Dict[str, datetime] = {}

    def update(self, emotion: str, confidence: float, source: str,
               last_message: str = "") -> dict:
        """
        Update student emotion state.

        Chat/voice messages bypass the smoothing history because text intent
        is explicit — if a student TYPES "I'm confused", that IS the ground truth.
        Face detection uses smoothing to avoid flickering.
        """
        if source == "chat" and emotion in ["confused", "angry", "frustrated", "anxious"]:
            # Chat is explicit — use directly, don't average with face history
            # But still add to history for future smoothing context
            self.emotion_history.append(emotion)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            smoothed = emotion   # ← trust the text directly
        else:
            # Face/screen: smooth with history to avoid flickering
            self.emotion_history.append(emotion)
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            from collections import Counter
            smoothed = Counter(self.emotion_history).most_common(1)[0][0]

        # Track duration of current emotion state
        if smoothed != self.emotion:
            self.emotion = smoothed
            self.current_emotion_start = datetime.now()

        self.duration_seconds = (datetime.now() - self.current_emotion_start).seconds
        self.confidence = confidence
        self.last_updated = datetime.now()
        self.source = source
        if last_message:
            self.last_message = last_message

        info = EMOTION_TO_STATE.get(smoothed, EMOTION_TO_STATE["neutral"])
        self.learning_state = info["state"]
        self.alert_level    = info["alert_level"]

        return self.to_dict()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "emotion": self.emotion,
            "emotion_emoji": EMOTION_EMOJI.get(self.emotion, "😐"),
            "confidence": round(self.confidence, 3),
            "learning_state": self.learning_state,
            "alert_level": self.alert_level,
            "alert_emoji": ALERT_EMOJI.get(self.alert_level, "🟢"),
            "duration_seconds": self.duration_seconds,
            "last_message": self.last_message,
            "source": self.source,
            "last_updated": self.last_updated.isoformat()
        }


class StudentTracker:
    def __init__(self):
        self._students: Dict[str, StudentState] = {}

    def add_student(self, name: str):
        if name not in self._students:
            self._students[name] = StudentState(name)

    def update_student_emotion(self, name: str, emotion: str, confidence: float,
                                source: str, last_message: str = "") -> dict:
        if name not in self._students:
            self.add_student(name)
        return self._students[name].update(emotion, confidence, source, last_message)

    def get_student(self, name: str) -> Optional[dict]:
        s = self._students.get(name)
        return s.to_dict() if s else None

    def get_all_students(self) -> List[dict]:
        return [s.to_dict() for s in self._students.values()]

    def get_class_stats(self) -> dict:
        if not self._students:
            return {"total": 0, "engaged": 0, "disengaged": 0,
                    "confused": 0, "engagement_percent": 0}

        states = [s.learning_state for s in self._students.values()]
        total = len(states)
        engaged = sum(1 for s in states if s in ["engaged", "curious", "motivated"])
        disengaged = sum(1 for s in states if s in ["disengaged", "averse"])
        confused = sum(1 for s in states if s in ["confused", "frustrated", "anxious"])
        neutral = total - engaged - disengaged - confused

        alert_students = [
            s.name for s in self._students.values() if s.alert_level >= 2
        ]

        return {
            "total": total,
            "engaged": engaged,
            "disengaged": disengaged,
            "confused": confused,
            "neutral": neutral,
            "engagement_percent": round((engaged / total) * 100) if total else 0,
            "alert_students": alert_students,
            "needs_attention_count": len(alert_students)
        }

    def student_count(self) -> int:
        return len(self._students)

    def get_student_state(self, name: str) -> Optional[StudentState]:
        return self._students.get(name)