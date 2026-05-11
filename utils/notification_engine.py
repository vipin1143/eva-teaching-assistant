"""
Notification Engine
Generates named per-student teacher alerts based on emotion thresholds.
Every notification includes the student's name, emotion, duration, and last message.
"""
from datetime import datetime, timedelta
from typing import List, Dict

# Thresholds before firing notification
THRESHOLDS = {
    "confused_seconds":    12,   # 12s confused → alert
    "frustrated_seconds":  10,
    "disengaged_seconds":  20,
    "anxious_seconds":     10,
    "class_wide_percent":  35,   # 35%+ class disengaged → urgent
}

COOLDOWN_SECONDS = 45  # Don't re-alert for same student+emotion within 45s

NOTIFICATION_TEMPLATES = {
    "confused": {
        "level": "high",
        "title": "{name} seems confused",
        "body": "Confused for {duration}s · Confidence {conf}%",
        "action": "Explain topic again or send hint",
        "color": "#f59e0b"
    },
    "frustrated": {
        "level": "high",
        "title": "{name} is frustrated",
        "body": "Frustrated for {duration}s · Consider simplifying",
        "action": "Break problem into smaller steps",
        "color": "#ef4444"
    },
    "disengaged": {
        "level": "medium",
        "title": "{name} is disengaged",
        "body": "Disengaged for {duration}s",
        "action": "Call on student or add interactive element",
        "color": "#8b5cf6"
    },
    "anxious": {
        "level": "high",
        "title": "{name} appears anxious",
        "body": "Anxious for {duration}s · Be encouraging",
        "action": "Reassure student, lower pressure",
        "color": "#f59e0b"
    },
    "chat_negative": {
        "level": "high",
        "title": "{name} sent a distress message",
        "body": '"{message}"',
        "action": "Respond to student's message",
        "color": "#ef4444"
    },
    "class_wide": {
        "level": "urgent",
        "title": "⚠️ Class-wide disengagement",
        "body": "{count} students disengaged: {names}",
        "action": "Consider a break or change teaching approach",
        "color": "#dc2626"
    },
    "re_engaged": {
        "level": "info",
        "title": "{name} is back on track",
        "body": "Returned to engaged state",
        "action": "",
        "color": "#10b981"
    }
}

CONFUSED_KEYWORDS = [
    "don't understand", "dont understand", "confused", "lost",
    "what does", "i don't get", "can you explain", "help",
    "not clear", "huh", "what is", "meaning", "stuck"
]

FRUSTRATED_KEYWORDS = [
    "this is hard", "too difficult", "give up", "can't do",
    "impossible", "hate this", "wrong again", "still wrong"
]


class NotificationEngine:
    def __init__(self):
        # student_name → {emotion: last_notified_at}
        self._last_notified: Dict[str, Dict[str, datetime]] = {}
        self._class_wide_cooldown: datetime = datetime.min

    def check_student(self, name: str, state: dict, tracker) -> List[dict]:
        """
        Check if a student's state warrants a teacher notification.
        Returns list of notification dicts (may be empty).
        """
        notifications = []
        emotion = state.get("emotion", "neutral")
        learning_state = state.get("learning_state", "neutral")
        duration = state.get("duration_seconds", 0)
        confidence = state.get("confidence", 0.5)
        last_message = state.get("last_message", "")

        # ── Keyword-based immediate alerts ───────────────────────
        if last_message:
            msg_lower = last_message.lower()
            if any(kw in msg_lower for kw in CONFUSED_KEYWORDS):
                if self._can_notify(name, "chat_negative"):
                    notifications.append(self._make_notif(
                        "chat_negative", name=name, message=last_message,
                        duration=duration, conf=int(confidence * 100)
                    ))
                    self._mark_notified(name, "chat_negative")

            elif any(kw in msg_lower for kw in FRUSTRATED_KEYWORDS):
                if self._can_notify(name, "chat_negative"):
                    notifications.append(self._make_notif(
                        "chat_negative", name=name, message=last_message,
                        duration=duration, conf=int(confidence * 100)
                    ))
                    self._mark_notified(name, "chat_negative")

        # ── Duration-based alerts ─────────────────────────────────
        if learning_state == "confused" and duration >= THRESHOLDS["confused_seconds"]:
            if self._can_notify(name, "confused"):
                notifications.append(self._make_notif(
                    "confused", name=name, duration=duration, conf=int(confidence * 100)
                ))
                self._mark_notified(name, "confused")

        elif learning_state == "frustrated" and duration >= THRESHOLDS["frustrated_seconds"]:
            if self._can_notify(name, "frustrated"):
                notifications.append(self._make_notif(
                    "frustrated", name=name, duration=duration, conf=int(confidence * 100)
                ))
                self._mark_notified(name, "frustrated")

        elif learning_state == "disengaged" and duration >= THRESHOLDS["disengaged_seconds"]:
            if self._can_notify(name, "disengaged"):
                notifications.append(self._make_notif(
                    "disengaged", name=name, duration=duration, conf=int(confidence * 100)
                ))
                self._mark_notified(name, "disengaged")

        elif learning_state == "anxious" and duration >= THRESHOLDS["anxious_seconds"]:
            if self._can_notify(name, "anxious"):
                notifications.append(self._make_notif(
                    "anxious", name=name, duration=duration, conf=int(confidence * 100)
                ))
                self._mark_notified(name, "anxious")

        # ── Re-engagement positive alert ──────────────────────────
        elif learning_state == "engaged":
            prev = self._last_notified.get(name, {})
            was_alerted = any(k in prev for k in ["confused", "frustrated", "disengaged"])
            if was_alerted and self._can_notify(name, "re_engaged"):
                notifications.append(self._make_notif("re_engaged", name=name,
                                                        duration=0, conf=0))
                self._mark_notified(name, "re_engaged")
                # Clear previous alert state
                self._last_notified[name] = {}

        # ── Class-wide check ─────────────────────────────────────
        class_stats = tracker.get_class_stats()
        disengaged_pct = (class_stats["disengaged"] / class_stats["total"] * 100
                          if class_stats["total"] > 0 else 0)

        if (disengaged_pct >= THRESHOLDS["class_wide_percent"] and
                self._can_notify("__class__", "class_wide")):
            alert_names = ", ".join(class_stats.get("alert_students", [])[:5])
            notifications.append(self._make_notif(
                "class_wide",
                name="Class",
                count=class_stats["disengaged"],
                names=alert_names,
                duration=0, conf=0
            ))
            self._mark_notified("__class__", "class_wide")

        return notifications

    def _make_notif(self, template_key: str, **kwargs) -> dict:
        tmpl = NOTIFICATION_TEMPLATES[template_key]
        return {
            "id": f"{kwargs.get('name', '')}_{template_key}_{int(datetime.now().timestamp())}",
            "type": template_key,
            "level": tmpl["level"],
            "title": tmpl["title"].format(**kwargs),
            "body": tmpl["body"].format(**kwargs),
            "action": tmpl["action"],
            "color": tmpl["color"],
            "student_name": kwargs.get("name", ""),
            "timestamp": datetime.now().isoformat(),
            "dismissed": False
        }

    def _can_notify(self, name: str, emotion: str) -> bool:
        last = self._last_notified.get(name, {}).get(emotion)
        if last is None:
            return True
        return (datetime.now() - last).seconds >= COOLDOWN_SECONDS

    def _mark_notified(self, name: str, emotion: str):
        if name not in self._last_notified:
            self._last_notified[name] = {}
        self._last_notified[name][emotion] = datetime.now()