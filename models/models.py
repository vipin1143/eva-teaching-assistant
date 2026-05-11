"""
models/models.py
SQLAlchemy ORM table definitions for EVA.
All database tables are declared here and imported by db_manager.py.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, Enum
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


# ── Base ──────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────
class EmotionSource(str, enum.Enum):
    face   = "face"     # Student link tab webcam
    screen = "screen"   # Teacher screen capture
    chat   = "chat"     # Zoom/Meet chat message


class AlertLevel(str, enum.Enum):
    info   = "info"
    medium = "medium"
    high   = "high"
    urgent = "urgent"


class LearningState(str, enum.Enum):
    engaged    = "engaged"
    neutral    = "neutral"
    confused   = "confused"
    frustrated = "frustrated"
    disengaged = "disengaged"
    anxious    = "anxious"
    curious    = "curious"
    averse     = "averse"


# ── Session ───────────────────────────────────────────────────────────────────
class Session(Base):
    """
    One row per live class session started by a teacher.
    """
    __tablename__ = "sessions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    teacher     = Column(String(100), nullable=False)
    class_name  = Column(String(100), nullable=False)
    subject     = Column(String(50),  default="general")
    started_at  = Column(DateTime,    default=datetime.now, nullable=False)
    ended_at    = Column(DateTime,    nullable=True)
    is_active   = Column(Boolean,     default=True)

    # Relationships
    students      = relationship("Student",          back_populates="session", cascade="all, delete-orphan")
    emotion_logs  = relationship("EmotionLog",       back_populates="session", cascade="all, delete-orphan")
    chat_logs     = relationship("ChatLog",          back_populates="session", cascade="all, delete-orphan")
    notifications = relationship("NotificationLog",  back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session id={self.id} class='{self.class_name}' teacher='{self.teacher}'>"


# ── Student ───────────────────────────────────────────────────────────────────
class Student(Base):
    """
    Every student who joins a session (via student link tab).
    Also created automatically when screen capture detects a new face.
    """
    __tablename__ = "students"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    name        = Column(String(100), nullable=False)
    joined_at   = Column(DateTime, default=datetime.now)
    source      = Column(String(20), default="student_link")  # student_link | screen_ocr | manual

    # Relationships
    session      = relationship("Session",    back_populates="students")
    emotion_logs = relationship("EmotionLog", back_populates="student_obj",
                                foreign_keys="EmotionLog.student_name",
                                primaryjoin="Student.name == EmotionLog.student_name",
                                viewonly=True)

    def __repr__(self):
        return f"<Student name='{self.name}' session={self.session_id}>"


# ── EmotionLog ────────────────────────────────────────────────────────────────
class EmotionLog(Base):
    """
    Timestamped emotion reading for one student.
    One row every ~3 seconds per student per source.
    """
    __tablename__ = "emotion_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    student_name = Column(String(100), nullable=False)   # denormalized for fast queries
    emotion      = Column(String(50),  nullable=False)
    confidence   = Column(Float,       default=0.5)
    learning_state = Column(String(50), default="neutral")
    source       = Column(Enum(EmotionSource), default=EmotionSource.face)
    logged_at    = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    session      = relationship("Session", back_populates="emotion_logs")

    def __repr__(self):
        return (f"<EmotionLog student='{self.student_name}' "
                f"emotion='{self.emotion}' source='{self.source}'>")


# ── ChatLog ───────────────────────────────────────────────────────────────────
class ChatLog(Base):
    """
    Every student chat message captured from Zoom webhook or OCR.
    Includes the sentiment result.
    """
    __tablename__ = "chat_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    student_name = Column(String(100), nullable=False)
    message      = Column(Text,        nullable=False)
    emotion      = Column(String(50),  default="neutral")   # sentiment result
    confidence   = Column(Float,       default=0.5)
    sent_at      = Column(DateTime,    default=datetime.now, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="chat_logs")

    def __repr__(self):
        return f"<ChatLog student='{self.student_name}' emotion='{self.emotion}'>"


# ── NotificationLog ───────────────────────────────────────────────────────────
class NotificationLog(Base):
    """
    Every alert sent to the teacher.
    Includes student name, reason, level and whether teacher dismissed it.
    """
    __tablename__ = "notification_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    student_name = Column(String(100), nullable=False)
    notif_type   = Column(String(50),  nullable=False)   # confused | frustrated | chat | classwide
    title        = Column(String(200), nullable=False)
    body         = Column(Text,        nullable=True)
    action       = Column(Text,        nullable=True)
    level        = Column(Enum(AlertLevel), default=AlertLevel.medium)
    dismissed    = Column(Boolean,     default=False)
    created_at   = Column(DateTime,    default=datetime.now, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="notifications")

    def __repr__(self):
        return (f"<NotificationLog student='{self.student_name}' "
                f"type='{self.notif_type}' level='{self.level}'>")


# ── LessonVariant ─────────────────────────────────────────────────────────────
class LessonVariant(Base):
    """
    Emotion-specific lesson content variants (EVA original feature).
    When a student is confused → serve simplified explanation.
    When engaged → serve challenge content.
    """
    __tablename__ = "lesson_variants"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    subject         = Column(String(50),  nullable=False)
    topic           = Column(String(100), nullable=False)
    emotion_trigger = Column(String(50),  nullable=False)   # confused | engaged | frustrated
    variant_type    = Column(String(30),  nullable=False)   # hint | challenge | simplification | encouragement
    content         = Column(Text,        nullable=False)
    difficulty      = Column(String(20),  default="medium") # easy | medium | hard
    created_at      = Column(DateTime,    default=datetime.now)

    def __repr__(self):
        return (f"<LessonVariant topic='{self.topic}' "
                f"trigger='{self.emotion_trigger}' type='{self.variant_type}'>")