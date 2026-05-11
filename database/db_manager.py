"""SQLAlchemy async database manager — concurrency-hardened for SQLite"""
import os
import asyncio
import logging
from datetime import datetime
from functools import wraps
from typing import List, Dict, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    select, func, update, event
)
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)
DB_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///database/eva.db")

os.makedirs("database", exist_ok=True)

# 🔧 FIX 1+2: SQLite concurrency settings
engine = create_async_engine(
    DB_URL,
    echo=False,
    connect_args={
        "timeout": 30,                  # wait up to 30s for lock
        "check_same_thread": False,
    },
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# 🔧 FIX 3: Retry decorator for "database is locked" errors
def with_db_retry(max_attempts: int = 5, base_delay: float = 0.1):
    """Retry a DB operation with exponential backoff if SQLite is locked."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except OperationalError as e:
                    msg = str(e).lower()
                    if "database is locked" in msg or "database table is locked" in msg:
                        last_exc = e
                        delay = base_delay * (2 ** attempt)
                        logger.debug(f"DB locked (attempt {attempt+1}/{max_attempts}), retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue
                    raise
            logger.error(f"DB still locked after {max_attempts} retries: {last_exc}")
            raise last_exc
        return wrapper
    return decorator


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    teacher     = Column(String(100))
    class_name  = Column(String(100))
    subject     = Column(String(50), default="general")
    started_at  = Column(DateTime, default=datetime.now)
    ended_at    = Column(DateTime, nullable=True)
    is_active   = Column(Boolean, default=True)


class Student(Base):
    __tablename__ = "students"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer)
    name        = Column(String(100))
    joined_at   = Column(DateTime, default=datetime.now)


class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer)
    student     = Column(String(100))
    emotion     = Column(String(50))
    confidence  = Column(Float)
    source      = Column(String(20))  # face / screen / chat
    logged_at   = Column(DateTime, default=datetime.now)


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer)
    student     = Column(String(100))
    message     = Column(Text)
    emotion     = Column(String(50))
    sent_at     = Column(DateTime, default=datetime.now)


class NotificationLog(Base):
    __tablename__ = "notification_logs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer)
    notif_type  = Column(String(50))
    title       = Column(String(200))
    body        = Column(Text)
    student     = Column(String(100))
    level       = Column(String(20))
    created_at  = Column(DateTime, default=datetime.now)
    dismissed   = Column(Boolean, default=False)


class DatabaseManager:
    def __init__(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._init_db())
        except RuntimeError:
            asyncio.run(self._init_db())
        except Exception as e:
            logger.warning(f"DB init deferred: {e}")

    async def _init_db(self):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # 🔧 FIX 2: Enable WAL mode + tuning PRAGMAs
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
            await conn.exec_driver_sql("PRAGMA busy_timeout=30000;")
            await conn.exec_driver_sql("PRAGMA cache_size=-64000;")  # 64MB cache
            await conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")
        logger.info("✅ Database ready (SQLAlchemy async, WAL mode enabled)")

    @with_db_retry()
    async def create_session(self, teacher: str, class_name: str, subject: str) -> int:
        async with AsyncSessionLocal() as s:
            obj = Session(teacher=teacher, class_name=class_name, subject=subject)
            s.add(obj)
            await s.commit()
            await s.refresh(obj)
            return obj.id

    @with_db_retry()
    async def end_session(self, session_id: int):
        async with AsyncSessionLocal() as s:
            obj = await s.get(Session, session_id)
            if obj:
                obj.ended_at = datetime.now()
                obj.is_active = False
                await s.commit()

    @with_db_retry()
    async def register_student(self, session_id: int, name: str):
        async with AsyncSessionLocal() as s:
            obj = Student(session_id=session_id, name=name)
            s.add(obj)
            await s.commit()

    @with_db_retry()
    async def log_emotion(self, session_id: int, student: str,
                          emotion: str, confidence: float, source: str):
        async with AsyncSessionLocal() as s:
            s.add(EmotionLog(
                session_id=session_id, student=student,
                emotion=emotion, confidence=confidence, source=source
            ))
            await s.commit()

    @with_db_retry()
    async def log_emotions_bulk(self, entries: List[dict]):
        """🆕 Bulk insert many emotion logs in one transaction (much faster).
        Each entry: {session_id, student, emotion, confidence, source}"""
        if not entries:
            return
        async with AsyncSessionLocal() as s:
            s.add_all([
                EmotionLog(
                    session_id=e["session_id"],
                    student=e["student"],
                    emotion=e["emotion"],
                    confidence=e["confidence"],
                    source=e.get("source", "face"),
                )
                for e in entries
            ])
            await s.commit()

    @with_db_retry()
    async def log_chat(self, session_id: int, student: str,
                       message: str, emotion: str):
        async with AsyncSessionLocal() as s:
            s.add(ChatLog(session_id=session_id, student=student,
                          message=message, emotion=emotion))
            await s.commit()

    @with_db_retry()
    async def log_notification(self, session_id: int, notif: dict):
        async with AsyncSessionLocal() as s:
            s.add(NotificationLog(
                session_id=session_id,
                notif_type=notif.get("type", ""),
                title=notif.get("title", ""),
                body=notif.get("body", ""),
                student=notif.get("student_name", ""),
                level=notif.get("level", "")
            ))
            await s.commit()

    @with_db_retry()
    async def get_notifications(self, session_id: int) -> List[dict]:
        async with AsyncSessionLocal() as s:
            rows = (await s.execute(
                select(NotificationLog)
                .where(NotificationLog.session_id == session_id)
                .order_by(NotificationLog.created_at.desc()).limit(50)
            )).scalars().all()
            return [{"id": r.id, "type": r.notif_type, "title": r.title,
                     "body": r.body, "student": r.student, "level": r.level,
                     "created_at": r.created_at.isoformat()} for r in rows]

    @with_db_retry()
    async def get_emotion_timeline(self, session_id: int, student: str) -> List[dict]:
        async with AsyncSessionLocal() as s:
            rows = (await s.execute(
                select(EmotionLog)
                .where(EmotionLog.session_id == session_id,
                       EmotionLog.student == student)
                .order_by(EmotionLog.logged_at.asc()).limit(200)
            )).scalars().all()
            return [{"emotion": r.emotion, "confidence": r.confidence,
                     "source": r.source, "time": r.logged_at.isoformat()} for r in rows]

    @with_db_retry()
    async def get_session_summary(self, session_id: int) -> dict:
        async with AsyncSessionLocal() as s:
            students = (await s.execute(
                select(func.count(Student.id))
                .where(Student.session_id == session_id)
            )).scalar()
            notifs = (await s.execute(
                select(func.count(NotificationLog.id))
                .where(NotificationLog.session_id == session_id)
            )).scalar()
            return {"total_students": students, "total_alerts": notifs}

    async def is_healthy(self) -> bool:
        try:
            async with AsyncSessionLocal() as s:
                await s.execute(select(func.now()))
            return True
        except Exception:
            return False

    @with_db_retry()
    async def session_exists(self, session_id: int) -> bool:
        """Check if a session ID exists and is still active."""
        async with AsyncSessionLocal() as s:
            result = await s.execute(
                select(Session).where(
                    Session.id == session_id,
                    Session.is_active == True
                )
            )
            return result.scalar_one_or_none() is not None

    @with_db_retry()
    async def close_all_sessions(self):
        """Mark ALL sessions as inactive — called on server startup."""
        async with AsyncSessionLocal() as s:
            await s.execute(
                update(Session)
                .where(Session.is_active == True)
                .values(is_active=False, ended_at=datetime.now())
            )
            await s.commit()
            logger.info("All stale sessions marked inactive")

    @with_db_retry()
    async def get_emotion_history_all(self, session_id: int) -> list:
        """Get all emotion logs for a session (all students)."""
        async with AsyncSessionLocal() as s:
            rows = (await s.execute(
                select(EmotionLog)
                .where(EmotionLog.session_id == session_id)
                .order_by(EmotionLog.logged_at.asc()).limit(1000)
            )).scalars().all()
            return [{"emotion": r.emotion, "confidence": r.confidence,
                     "student": r.student, "source": r.source,
                     "time": r.logged_at.isoformat()} for r in rows]