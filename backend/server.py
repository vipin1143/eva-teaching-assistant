"""
FastAPI Server – All API routes + WebSocket for live teacher dashboard
"""
import os, json, asyncio, logging, base64, hashlib, hmac
from datetime import datetime
from typing import List, Set, Optional, Dict
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from utils.emotion_detector import EmotionDetector
from utils.text_sentiment import TextSentimentAnalyzer
from utils.student_tracker import StudentTracker
from utils.notification_engine import NotificationEngine
from utils.screen_capture import ScreenCapture
from utils.ai_suggestion import AISuggestion
from utils.session_report import generate_report_html
from database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class SessionStart(BaseModel):
    teacher_name: str
    class_name: str
    subject: str = "general"

class StudentFrame(BaseModel):
    student_id: str
    student_name: str
    image_b64: str
    session_id: int

class ChatMessage(BaseModel):
    student_name: str
    message: str
    session_id: int
    platform: str = "eva"       # eva | zoom | meet

class StudentJoin(BaseModel):
    student_name: str
    session_id: int

class EmotionFrame(BaseModel):
    image_b64: str   # full Zoom/Meet screen
    session_id: int
    page_number: int = 1


# ── 🆕 Buffered Emotion Logger ───────────────────────────────────────────────
class EmotionBuffer:
    """Collects emotion logs in memory; flushes in bulk every FLUSH_INTERVAL seconds.
    Eliminates SQLite write contention from per-frame inserts."""
    FLUSH_INTERVAL = 5.0  # seconds
    MAX_BUFFER_SIZE = 500  # force-flush if we hit this many entries

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._buf: List[dict] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info(f"✅ Emotion buffer started (flush every {self.FLUSH_INTERVAL}s)")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def add(self, session_id: int, student: str,
                  emotion: str, confidence: float, source: str):
        force_flush = False
        async with self._lock:
            self._buf.append({
                "session_id": session_id,
                "student": student,
                "emotion": emotion,
                "confidence": confidence,
                "source": source,
            })
            if len(self._buf) >= self.MAX_BUFFER_SIZE:
                force_flush = True
        if force_flush:
            await self.flush()

    async def flush(self):
        async with self._lock:
            if not self._buf:
                return
            snapshot = self._buf
            self._buf = []
        try:
            await self.db.log_emotions_bulk(snapshot)
            logger.debug(f"💾 Flushed {len(snapshot)} emotion logs")
        except Exception as e:
            logger.error(f"Buffer flush failed (re-queueing): {e}")
            # On failure, re-queue so we don't lose data
            async with self._lock:
                self._buf = snapshot + self._buf

    async def _loop(self):
        try:
            while True:
                await asyncio.sleep(self.FLUSH_INTERVAL)
                await self.flush()
        except asyncio.CancelledError:
            logger.info("Emotion buffer loop cancelled")
            raise


# ── WebSocket Connection Manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self.live_sessions: Set[int] = set()   # sessions started THIS server run

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    def register_session(self, session_id: int):
        self.live_sessions.add(session_id)

    def close_session(self, session_id: int):
        self.live_sessions.discard(session_id)

    def is_live_session(self, session_id: int) -> bool:
        return session_id in self.live_sessions

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.active -= dead


def create_app() -> FastAPI:
    app = FastAPI(title="EVA Teaching Assistant", version="2.0.0")

    app.add_middleware(CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Init services
    db = DatabaseManager()
    detector = EmotionDetector()
    sentiment = TextSentimentAnalyzer()
    tracker = StudentTracker()
    notifier = NotificationEngine()
    capture = ScreenCapture()
    manager = ConnectionManager()
    ai_suggest = AISuggestion()
    emotion_buffer = EmotionBuffer(db)            # 🆕
    session_start_times: Dict[int, datetime] = {} # session_id → start datetime
    session_meta: Dict[int, dict] = {}            # 🆕 keep class/subject/teacher for reports

    # Static files
    base = os.path.dirname(__file__)
    teacher_path = os.path.join(base, "../teacher_dashboard")
    student_path = os.path.join(base, "../student_link")
    if os.path.exists(teacher_path):
        app.mount("/teacher_static", StaticFiles(directory=teacher_path), name="teacher")
    if os.path.exists(student_path):
        app.mount("/student_static", StaticFiles(directory=student_path), name="student")

    # ── Page Routes ───────────────────────────────────────────────
    @app.get("/teacher", response_class=HTMLResponse)
    async def teacher_dashboard():
        p = os.path.join(base, "../teacher_dashboard/index.html")
        return FileResponse(p)

    @app.get("/student", response_class=HTMLResponse)
    async def student_page():
        p = os.path.join(base, "../student_link/index.html")
        return FileResponse(p)

    # ── WebSocket (live teacher feed) ─────────────────────────────
    @app.websocket("/ws/teacher/{session_id}")
    async def teacher_ws(websocket: WebSocket, session_id: int):
        await manager.connect(websocket)
        logger.info(f"Teacher WS connected for session {session_id}")
        try:
            while True:
                await websocket.receive_text()  # keep alive ping
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # ── Session Management ────────────────────────────────────────
    @app.post("/api/session/start")
    async def start_session(data: SessionStart):
        sid = await db.create_session(data.teacher_name, data.class_name, data.subject)
        manager.register_session(sid)
        session_start_times[sid] = datetime.now()
        session_meta[sid] = {
            "class_name": data.class_name,
            "subject":    data.subject,
            "teacher":    data.teacher_name,
        }
        logger.info(f"✅ Live session #{sid} registered for {data.class_name}")
        return {"success": True, "session_id": sid,
                "message": f"Session started for {data.class_name}",
                "class_name": data.class_name,
                "subject": data.subject,
                "teacher": data.teacher_name}

    @app.post("/api/session/{session_id}/end")
    async def end_session(session_id: int):
        # 🆕 Flush buffer FIRST so ending session has all data
        await emotion_buffer.flush()
        summary = await db.get_session_summary(session_id)
        await db.end_session(session_id)
        manager.close_session(session_id)
        capture.stop()
        return {"success": True, "summary": summary}

    # ── Student Registration ──────────────────────────────────────
    @app.post("/api/student/join")
    async def student_join(data: StudentJoin):
        """Called when student opens the student link page"""
        if not manager.is_live_session(data.session_id):
            return {
                "success": False,
                "message": f"Session #{data.session_id} is not active. "
                           f"Please check the Session ID with your teacher."
            }

        await db.register_student(data.session_id, data.student_name)
        tracker.add_student(data.student_name)
        await manager.broadcast({
            "type": "student_joined",
            "student_name": data.student_name,
            "timestamp": datetime.now().isoformat()
        })
        return {"success": True, "message": f"Welcome {data.student_name}!"}

    # ── Student Frame (from student link tab) ─────────────────────
    @app.post("/api/emotion/student-frame")
    async def process_student_frame(data: StudentFrame):
        """
        Student's own browser sends their webcam frame every 3s.
        No Zoom access needed – works for ALL students including off-screen.
        """
        result = detector.detect_from_base64(data.image_b64, smooth_key=data.student_name)
        emotion    = result.get("emotion", "neutral")
        confidence = result.get("confidence", 0.5)
        face_found = result.get("face_found", False)

        logger.info(
            f"Frame from {data.student_name}: "
            f"face_found={face_found} emotion={emotion} conf={confidence:.2f} "
            f"method={result.get('method','?')}"
        )

        # Update tracker (in-memory, fast)
        updated = tracker.update_student_emotion(
            data.student_name, emotion, confidence, source="face"
        )

        # 🆕 Buffer the DB write instead of writing immediately
        await emotion_buffer.add(
            data.session_id, data.student_name, emotion, confidence, "face"
        )

        # Check notification thresholds
        notifications = notifier.check_student(data.student_name, updated, tracker)
        for notif in notifications:
            await manager.broadcast({"type": "notification", **notif})
            await db.log_notification(data.session_id, notif)

        # Broadcast updated student state WITH face image (live)
        await manager.broadcast({
            "type": "student_emotion_update",
            "student_name": data.student_name,
            "emotion": emotion,
            "confidence": confidence,
            "learning_state": updated.get("learning_state"),
            "duration_seconds": updated.get("duration_seconds", 0),
            "face_image": data.image_b64,
            "face_found": face_found,
            "timestamp": datetime.now().isoformat()
        })

        return {"success": True, "emotion": emotion}

    # ── Screen Capture (teacher machine, visible tiles only) ──────
    @app.post("/api/emotion/screen-frame")
    async def process_screen_frame(data: EmotionFrame):
        faces = detector.detect_all_faces_from_screen(data.image_b64)
        results = []

        for face in faces:
            name = face.get("name_ocr", f"Student_{face['tile_index']}")
            emotion = face.get("emotion", "neutral")
            confidence = face.get("confidence", 0.5)

            updated = tracker.update_student_emotion(
                name, emotion, confidence, source="screen"
            )
            # 🆕 Buffer instead of direct write
            await emotion_buffer.add(
                data.session_id, name, emotion, confidence, "screen"
            )

            notifications = notifier.check_student(name, updated, tracker)
            for notif in notifications:
                await manager.broadcast({"type": "notification", **notif})

            results.append({"name": name, "emotion": emotion})

        class_stats = tracker.get_class_stats()
        await manager.broadcast({"type": "class_stats_update", **class_stats})

        return {"success": True, "faces_detected": len(faces), "results": results}

    # ── Chat Message Analysis ─────────────────────────────────────
    @app.post("/api/chat/analyze")
    async def analyze_chat(data: ChatMessage):
        result = sentiment.analyze(data.message)
        emotion = result["emotion"]
        confidence = result["confidence"]

        updated = tracker.update_student_emotion(
            data.student_name, emotion, confidence, source="chat",
            last_message=data.message
        )
        # 🆕 Buffer the emotion log; chat log stays direct (less frequent)
        await emotion_buffer.add(
            data.session_id, data.student_name, emotion, confidence, "chat"
        )
        await db.log_chat(data.session_id, data.student_name, data.message, emotion)

        notifications = notifier.check_student(data.student_name, updated, tracker)
        for notif in notifications:
            await manager.broadcast({"type": "notification", **notif})

        await manager.broadcast({
            "type": "chat_emotion",
            "student_name": data.student_name,
            "message": data.message,
            "emotion": emotion,
            "learning_state": updated.get("learning_state", emotion),
            "alert_level": updated.get("alert_level", 0),
            "confidence": confidence,
            "duration_seconds": updated.get("duration_seconds", 0),
            "source": "chat",
            "timestamp": datetime.now().isoformat()
        })

        # Instant alert for help/confused keywords
        help_keywords = [
            "help", "confused", "don't understand", "dont understand",
            "explain", "lost", "not clear", "i need help", "i'm confused",
            "can you explain", "don't get", "dont get"
        ]
        urgent_keywords = [
            "i need help", "i'm confused", "don't understand",
            "can you explain", "explain again", "not clear"
        ]
        msg_lower = data.message.lower()
        if any(kw in msg_lower for kw in help_keywords):
            level = "urgent" if any(kw in msg_lower for kw in urgent_keywords) else "high"
            instant_notif = {
                "type": "notification",
                "id": f"{data.student_name}_chat_{int(datetime.now().timestamp())}",
                "level": level,
                "title": f"💬 {data.student_name} needs help",
                "body": f'"{data.message}"',
                "action": "Respond to this student directly",
                "color": "#ff4d6d",
                "student_name": data.student_name,
                "timestamp": datetime.now().isoformat(),
                "dismissed": False
            }
            await manager.broadcast(instant_notif)
            await db.log_notification(data.session_id, instant_notif)

        return {"success": True, **result}

    # ── Zoom Webhook ─────────────────────────────────────────────
    @app.api_route("/api/zoom/webhook", methods=["GET", "POST", "HEAD"])
    async def zoom_webhook(
        request: Request,
        x_zm_signature: Optional[str] = Header(None),
        x_zm_request_timestamp: Optional[str] = Header(None),
    ):
        if request.method in ("GET", "HEAD"):
            logger.info(f"✅ Zoom {request.method} check received — OK")
            return JSONResponse({"status": "ok", "service": "EVA Teaching Assistant"})

        raw_body    = await request.body()
        zoom_secret = os.getenv("ZOOM_WEBHOOK_SECRET", "")

        try:
            payload = json.loads(raw_body) if raw_body else {}
        except Exception:
            payload = {}

        event = payload.get("event", "")
        logger.info(f"📨 Zoom webhook event: '{event}'")

        if event == "endpoint.url_validation":
            plain_token = payload.get("payload", {}).get("plainToken", "")
            logger.info(f"🔐 Zoom URL validation — plainToken: {plain_token[:10]}...")
            if zoom_secret and plain_token:
                encrypted = hmac.new(
                    zoom_secret.encode(), plain_token.encode(), hashlib.sha256
                ).hexdigest()
            else:
                encrypted = plain_token
            logger.info("✅ Zoom URL validation response sent")
            return JSONResponse({
                "plainToken":     plain_token,
                "encryptedToken": encrypted
            })

        if zoom_secret and x_zm_signature and x_zm_request_timestamp:
            msg_to_hash = f"v0:{x_zm_request_timestamp}:{raw_body.decode()}"
            expected    = "v0=" + hmac.new(
                zoom_secret.encode(), msg_to_hash.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(x_zm_signature, expected):
                logger.warning("⚠️  Zoom signature mismatch")
                raise HTTPException(status_code=401, detail="Invalid signature")

        active_sessions = list(manager.live_sessions)
        if not active_sessions:
            logger.warning("Zoom webhook: no active session to attach to")
            return {"status": "no_active_session"}
        session_id = max(active_sessions)

        if event == "chat_message.sent":
            obj     = payload.get("payload", {}).get("object", {})
            sender  = obj.get("sender",  {}).get("display_name", "Unknown Student")
            message = obj.get("message", "").strip()

            if not message:
                return {"status": "empty_message"}

            logger.info(f"💬 Zoom chat from {sender}: {message}")

            if sender not in [s["name"] for s in tracker.get_all_students()]:
                tracker.add_student(sender)
                await db.register_student(session_id, sender)
                await manager.broadcast({
                    "type": "student_joined",
                    "student_name": sender,
                    "source": "zoom_chat",
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"👋 Auto-registered Zoom student: {sender}")

            await analyze_chat(ChatMessage(
                student_name=sender, message=message, session_id=session_id
            ))

        elif event == "meeting.participant_joined":
            participant = payload.get("payload", {}).get("object", {}).get("participant", {})
            name = participant.get("user_name", "Unknown Student")
            if name and name != "Unknown Student":
                tracker.add_student(name)
                await db.register_student(session_id, name)
                await manager.broadcast({
                    "type": "student_joined",
                    "student_name": name,
                    "source": "zoom_join",
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"👋 Student joined Zoom: {name}")

        elif event == "meeting.participant_left":
            participant = payload.get("payload", {}).get("object", {}).get("participant", {})
            name = participant.get("user_name", "")
            if name:
                await manager.broadcast({
                    "type": "student_left",
                    "student_name": name,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"👋 Student left Zoom: {name}")

        return {"status": "ok", "event": event}

    @app.get("/api/zoom/webhook-url")
    async def get_webhook_url():
        try:
            import requests as req
            tunnels = req.get("http://localhost:4040/api/tunnels", timeout=2).json()
            for t in tunnels.get("tunnels", []):
                if t.get("proto") == "https":
                    url = t["public_url"]
                    return {
                        "ngrok_url": url,
                        "webhook_endpoint": f"{url}/api/zoom/webhook",
                        "instructions": (
                            f"Paste this in Zoom App Marketplace → "
                            f"Event Subscriptions → Endpoint URL: "
                            f"{url}/api/zoom/webhook"
                        )
                    }
        except Exception:
            pass
        return {
            "ngrok_url": None,
            "message": "ngrok not running. Start with: ngrok http 8000",
            "webhook_endpoint": "https://YOUR_NGROK_URL/api/zoom/webhook"
        }

    # ════════════════════════════════════════════════════════════
    # GOOGLE MEET INTEGRATION
    # ════════════════════════════════════════════════════════════
    @app.post("/api/meet/webhook")
    async def meet_webhook(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        event_type = payload.get("type", "")
        logger.info(f"📨 Google Meet event: {event_type}")

        active_sessions = list(manager.live_sessions)
        if not active_sessions:
            logger.warning("Meet webhook: no active EVA session")
            return {"text": "No active EVA session. Teacher must start a session first."}
        session_id = max(active_sessions)

        if event_type == "MESSAGE":
            sender_obj = payload.get("message", {}).get("sender", {})
            sender     = sender_obj.get("displayName", "Unknown Student")
            message    = payload.get("message", {}).get("text", "").strip()

            if not message or sender_obj.get("type") == "BOT":
                return {"text": ""}

            logger.info(f"💬 Meet chat from {sender}: {message}")

            known = [s["name"] for s in tracker.get_all_students()]
            if sender not in known:
                tracker.add_student(sender)
                await db.register_student(session_id, sender)
                await manager.broadcast({
                    "type": "student_joined",
                    "student_name": sender,
                    "source": "meet_chat",
                    "timestamp": datetime.now().isoformat()
                })

            await analyze_chat(ChatMessage(
                student_name=sender, message=message,
                session_id=session_id, platform="meet"
            ))
            return {"text": ""}

        elif event_type == "ADDED_TO_SPACE":
            space_type = payload.get("space", {}).get("type", "")
            if space_type == "ROOM":
                logger.info("✅ EVA bot added to Google Meet space")
                return {"text": "EVA is now monitoring this class session. 📊"}
            else:
                sender = payload.get("user", {}).get("displayName", "")
                if sender:
                    tracker.add_student(sender)
                    await db.register_student(session_id, sender)
                    await manager.broadcast({
                        "type": "student_joined",
                        "student_name": sender,
                        "source": "meet_join",
                        "timestamp": datetime.now().isoformat()
                    })
                return {"text": ""}

        elif event_type == "REMOVED_FROM_SPACE":
            sender = payload.get("user", {}).get("displayName", "")
            if sender:
                await manager.broadcast({
                    "type": "student_left",
                    "student_name": sender,
                    "timestamp": datetime.now().isoformat()
                })
            return {"text": ""}

        return {"text": ""}

    @app.post("/api/meet/apps-script")
    async def meet_apps_script(request: Request):
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        student_name = data.get("student_name", "Unknown Student")
        message      = data.get("message", "").strip()

        if not message:
            return {"status": "empty"}

        active_sessions = list(manager.live_sessions)
        if not active_sessions:
            return {"status": "no_active_session"}
        session_id = max(active_sessions)

        logger.info(f"💬 Meet Apps Script from {student_name}: {message}")

        known = [s["name"] for s in tracker.get_all_students()]
        if student_name not in known:
            tracker.add_student(student_name)
            await db.register_student(session_id, student_name)
            await manager.broadcast({
                "type": "student_joined",
                "student_name": student_name,
                "source": "meet_script",
                "timestamp": datetime.now().isoformat()
            })

        await analyze_chat(ChatMessage(
            student_name=student_name, message=message,
            session_id=session_id, platform="meet"
        ))
        return {"status": "ok", "student": student_name}

    @app.get("/api/meet/webhook-url")
    async def get_meet_webhook_url():
        try:
            import requests as req
            tunnels = req.get("http://localhost:4040/api/tunnels", timeout=2).json()
            for t in tunnels.get("tunnels", []):
                if t.get("proto") == "https":
                    url = t["public_url"]
                    return {
                        "ngrok_url": url,
                        "chat_api_endpoint":    f"{url}/api/meet/webhook",
                        "apps_script_endpoint": f"{url}/api/meet/apps-script",
                        "recommended": "apps_script — easiest setup, no Google Cloud needed"
                    }
        except Exception:
            pass
        return {
            "ngrok_url": None,
            "message": "ngrok not running. Start: ngrok http 8000"
        }

    # ── Progress & Stats ──────────────────────────────────────────
    @app.get("/api/session/{session_id}/stats")
    async def get_session_stats(session_id: int):
        students = tracker.get_all_students()
        class_stats = tracker.get_class_stats()
        return {"success": True, "students": students, "class_stats": class_stats}

    @app.get("/api/session/{session_id}/notifications")
    async def get_notifications(session_id: int):
        notifs = await db.get_notifications(session_id)
        return {"success": True, "notifications": notifs}

    @app.get("/api/session/{session_id}/emotion-timeline/{student_name}")
    async def get_emotion_timeline(session_id: int, student_name: str):
        timeline = await db.get_emotion_timeline(session_id, student_name)
        return {"success": True, "timeline": timeline}

    # ── Screen Capture Control ────────────────────────────────────
    @app.post("/api/capture/start")
    async def start_capture(session_id: int):
        capture.start(session_id)
        return {"success": True, "message": "Screen capture started"}

    @app.post("/api/capture/stop")
    async def stop_capture():
        capture.stop()
        return {"success": True}

    # ── Feature 1: AI Suggestion ──────────────────────────────────
    @app.post("/api/suggestion")
    async def get_suggestion(data: dict):
        result = ai_suggest.get_suggestion(
            student_name=data.get("student_name", "Student"),
            emotion=data.get("emotion", "confused"),
            message=data.get("message", ""),
            subject=data.get("subject", "general")
        )
        return {"success": True, **result}

    # ── Feature 3: Session Report ──────────────────────────────────
    @app.get("/api/session/{session_id}/report")
    async def get_session_report(session_id: int):
        """Generate full HTML session report."""
        # 🆕 Flush buffer first so report includes most recent data
        await emotion_buffer.flush()

        from fastapi.responses import HTMLResponse as HR
        students_data = tracker.get_all_students()
        emotion_logs  = await db.get_emotion_history_all(session_id)
        notifs        = await db.get_notifications(session_id)
        summary       = await db.get_session_summary(session_id)

        start_time = session_start_times.get(session_id, datetime.now())
        duration   = int((datetime.now() - start_time).total_seconds())

        # 🆕 Pull from in-memory meta (was crashing before — keys didn't exist on summary)
        meta = session_meta.get(session_id, {})
        session_info = {
            "class_name": meta.get("class_name", "Class"),
            "subject":    meta.get("subject", "General"),
            "teacher":    meta.get("teacher", "Teacher"),
        }

        html = generate_report_html(
            session_info=session_info,
            students=students_data,
            emotion_logs=emotion_logs,
            notifications=notifs,
            duration_seconds=duration
        )
        return HR(content=html)

    # ── Feature 6: Multi-language sentiment ───────────────────────
    @app.post("/api/translate-analyze")
    async def translate_and_analyze(data: dict):
        message = data.get("message", "")
        result = sentiment.analyze(message)
        return {"success": True, **result}

    # ── Lifecycle Hooks ───────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        """Mark all old DB sessions as inactive on server restart + start buffer."""
        await db.close_all_sessions()
        emotion_buffer.start()                         # 🆕
        logger.info("✅ All old sessions marked inactive on startup")

    @app.on_event("shutdown")
    async def on_shutdown():
        """🆕 Flush emotion buffer before shutdown so no data is lost."""
        await emotion_buffer.stop()
        logger.info("✅ Emotion buffer drained on shutdown")

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "services": {
                "emotion_detector": detector.is_ready(),
                "text_sentiment": sentiment.is_ready(),
                "database": await db.is_healthy(),
                "active_students": tracker.student_count(),
                "live_sessions": list(manager.live_sessions),
                "buffer_size": len(emotion_buffer._buf),  # 🆕 visibility
            }
        }

    return app