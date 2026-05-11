# EVA – Emotion-Adaptive Virtual Teaching Assistant v2.0

**100% teacher-side. Students just join Zoom/Meet normally.**

---

## 🏗️ Architecture (Hybrid for 70+ students)

```
Students (70) ──┬── Student Link Tab  ──► Face frames every 3s  ──► ONNX FER+
                ├── Zoom/Meet Chat    ──► Zoom Webhook / OCR    ──► VADER Sentiment  
                └── Screen Capture   ──► Visible tile grid      ──► Multi-face detect

                        All feeds → Emotion Fusion → Per-student named alerts
                                                   → Teacher Dashboard (WebSocket live)
```

| Source | Covers | How |
|---|---|---|
| Student Link Tab | ALL 70 students | Student opens 1 URL, camera sends frames |
| Zoom Chat webhook | ALL 70 students | Zoom sends every chat message to backend |
| Screen capture | ~25 visible on screen | Auto page-cycles every 12s |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ONNX model (~30MB)
python models/download_models.py

# 3. Add API key to .env
ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python app.py

# 5. Open teacher dashboard
# http://localhost:8000/teacher

# 6. Share student link (they just open it once)
# http://localhost:8000/student
```

---

## 🔔 Named Notifications

Every alert includes the exact student name:

```
🔴 Rahul Verma is confused
   Confused for 45s · Face detection · 78% confidence
   👉 Explain the current topic again or send a hint
   
🟡 Priya Singh seems disengaged  
   Disengaged for 30s · No recent chat
   👉 Call on student or add interactive element

💬 Amit Sharma sent a distress message
   "i dont understand this at all"
   👉 Respond directly to this student's message

⚠️ Multiple students disengaged
   Rahul, Priya, Sara, Dev and others need help
   👉 Consider pausing and recapping the topic
```

---

## 📁 Project Structure

```
eva_project/
├── app.py                      ← Entry point (uvicorn)
├── requirements.txt
├── .env
├── backend/
│   └── server.py               ← FastAPI routes + WebSocket
├── utils/
│   ├── emotion_detector.py     ← ONNX FER+ + multi-face grid
│   ├── text_sentiment.py       ← VADER + keyword rules  
│   ├── student_tracker.py      ← Per-student emotion state
│   ├── notification_engine.py  ← Named alert generation
│   └── screen_capture.py      ← MSS capture + page cycling
├── database/
│   └── db_manager.py           ← SQLAlchemy async (SQLite/Postgres)
├── models/
│   ├── download_models.py      ← Model downloader
│   ├── emotion_ferplus_8.onnx  ← (downloaded)
│   └── haarcascade_*.xml       ← (downloaded)
├── teacher_dashboard/
│   └── index.html              ← Live teacher UI
└── student_link/
    └── index.html              ← Student one-click join page
```

---

## 🔌 Zoom Webhook Setup

1. Go to https://marketplace.zoom.us → Create App → Webhook Only
2. Event Subscriptions → Add: `chat.message.sent`
3. Set endpoint URL: `https://your-server/api/zoom/webhook`
4. Copy webhook secret to `.env` as `ZOOM_WEBHOOK_SECRET`

---

## ⚙️ Notification Thresholds (configurable in `notification_engine.py`)

| Trigger | Threshold | Alert Level |
|---|---|---|
| Confused (face/chat) | 12 seconds | 🔴 High |
| Frustrated | 10 seconds | 🔴 High |
| Disengaged | 20 seconds | 🟡 Medium |
| Chat distress keywords | Instant | 🔴 High |
| Class-wide (35%+ disengaged) | Instant | ⚠️ Urgent |
| Re-engagement | Instant | ✅ Info |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, uvicorn |
| Emotion (face) | ONNX Runtime + FER+ model |
| Emotion (text) | VADER Sentiment + keyword rules |
| Screen capture | MSS (fast cross-platform) |
| Name reading | Pytesseract OCR |
| Page cycling | PyAutoGUI |
| Database | SQLAlchemy async + SQLite/PostgreSQL |
| Real-time | WebSocket (FastAPI native) |
| Teacher UI | HTML5 + Vanilla JS |
| Student page | Single HTML page (no framework) |