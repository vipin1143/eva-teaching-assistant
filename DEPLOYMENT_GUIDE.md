# 🚀 EVA Deployment Guide

## Overview

This guide covers deploying the EVA (Emotion-Adaptive Virtual Assistant) application to production.

---

## 📋 Pre-Deployment Checklist

### ✅ **1. Test Locally**
- [x] Emotion detection working (happy, sad, neutral, etc.)
- [x] Student webcam feed working
- [x] Teacher dashboard displaying emotions
- [x] Chat sentiment analysis working
- [x] Database saving session data
- [x] All models loaded successfully

### ✅ **2. Environment Variables**
Check your `.env` file has all required variables:

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./database/eva.db

# AI/LLM (for suggestions)
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile

# Zoom Integration (optional)
ZOOM_WEBHOOK_SECRET=your_zoom_secret_here

# Google Meet Integration (optional)
GOOGLE_MEET_WEBHOOK_URL=your_meet_webhook_url
```

### ✅ **3. Required Files**
Ensure these model files exist:

```
models/
├── eva_emotion_v1.onnx              ✅ Primary emotion model
├── emotion_ferplus_8.onnx           ✅ Fallback model
├── haarcascade_frontalface_default.xml  ✅ Face detection
├── face_landmarker.tflite           ✅ Head pose
└── shape_predictor_68_face_landmarks.dat ✅ Face alignment

yolov8n-face.pt                      ✅ YOLO face detector
```

---

## 🌐 Deployment Options

### **Option 1: Cloud Platform (Recommended)**

#### **A. Render.com (Free Tier Available)**

1. **Create `render.yaml`:**
```yaml
services:
  - type: web
    name: eva-teaching-assistant
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 8000
      - key: LLM_API_KEY
        sync: false
```

2. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/eva-project.git
git push -u origin main
```

3. **Deploy on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect Python and deploy

4. **Set Environment Variables:**
   - In Render dashboard → Environment
   - Add `LLM_API_KEY` and other secrets

**Pros:** Free tier, auto-deploy from GitHub, HTTPS included
**Cons:** Cold starts (app sleeps after 15 min inactivity on free tier)

---

#### **B. Railway.app**

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

2. **Login and Deploy:**
```bash
railway login
railway init
railway up
```

3. **Set Environment Variables:**
```bash
railway variables set LLM_API_KEY=your_key_here
```

**Pros:** Easy deployment, good free tier, fast
**Cons:** Limited free hours per month

---

#### **C. Heroku**

1. **Create `Procfile`:**
```
web: python app.py
```

2. **Create `runtime.txt`:**
```
python-3.11.0
```

3. **Deploy:**
```bash
heroku login
heroku create eva-teaching-assistant
git push heroku main
heroku config:set LLM_API_KEY=your_key_here
```

**Pros:** Mature platform, good documentation
**Cons:** No free tier anymore (paid only)

---

#### **D. Google Cloud Run**

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "app.py"]
```

2. **Deploy:**
```bash
gcloud run deploy eva-assistant \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Pros:** Scales to zero (pay only when used), fast, reliable
**Cons:** Requires Google Cloud account, more complex setup

---

#### **E. AWS EC2 (Traditional Server)**

1. **Launch EC2 Instance:**
   - Ubuntu 22.04 LTS
   - t2.medium or larger (for model inference)
   - Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **SSH into server:**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3.11 python3-pip nginx -y
```

4. **Clone and setup:**
```bash
git clone https://github.com/yourusername/eva-project.git
cd eva-project
pip3 install -r requirements.txt
```

5. **Create systemd service (`/etc/systemd/system/eva.service`):**
```ini
[Unit]
Description=EVA Teaching Assistant
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/eva-project
Environment="PATH=/home/ubuntu/.local/bin"
ExecStart=/usr/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

6. **Start service:**
```bash
sudo systemctl enable eva
sudo systemctl start eva
```

7. **Configure Nginx (`/etc/nginx/sites-available/eva`):**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

8. **Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/eva /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Pros:** Full control, no cold starts, can handle heavy load
**Cons:** More expensive, requires server management

---

### **Option 2: Local Network Deployment**

For use within a school/organization network:

1. **Find your local IP:**
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

2. **Update app.py to bind to all interfaces:**
```python
# Already configured:
uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
```

3. **Share the URL:**
```
Teacher Dashboard: http://YOUR_LOCAL_IP:8000/teacher
Student Link: http://YOUR_LOCAL_IP:8000/student
```

4. **Keep computer running:**
   - Disable sleep mode
   - Ensure stable power supply
   - Keep terminal/command prompt open

**Pros:** Free, fast, no internet required
**Cons:** Only works on local network, computer must stay on

---

## 🔒 Security Considerations

### **1. HTTPS/SSL**

For production, always use HTTPS:

**Option A: Let's Encrypt (Free)**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

**Option B: Cloudflare (Free)**
- Add your domain to Cloudflare
- Enable "Full" SSL mode
- Cloudflare provides free SSL certificate

### **2. Environment Variables**

Never commit `.env` to Git:
```bash
# Add to .gitignore
echo ".env" >> .gitignore
```

### **3. Database Backups**

Backup SQLite database regularly:
```bash
# Backup script
cp database/eva.db database/backups/eva_$(date +%Y%m%d_%H%M%S).db
```

### **4. Rate Limiting**

Add rate limiting to prevent abuse:
```python
# In backend/server.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/emotion/student-frame")
@limiter.limit("30/minute")  # 30 requests per minute
async def process_student_frame(request: Request, data: StudentFrame):
    # ... existing code
```

---

## 📊 Monitoring & Maintenance

### **1. Logging**

Check logs regularly:
```bash
# If using systemd
sudo journalctl -u eva -f

# If using Docker
docker logs -f eva-container

# If running directly
# Logs are in terminal output
```

### **2. Database Maintenance**

Clean old sessions periodically:
```python
# Add to backend/server.py
@app.post("/api/admin/cleanup")
async def cleanup_old_sessions():
    # Delete sessions older than 30 days
    await db.cleanup_old_sessions(days=30)
    return {"success": True}
```

### **3. Model Updates**

To update emotion models:
```bash
# Backup old model
cp models/eva_emotion_v1.onnx models/eva_emotion_v1_backup.onnx

# Replace with new model
cp new_model.onnx models/eva_emotion_v1.onnx

# Restart server
sudo systemctl restart eva
```

---

## 🔧 Production Optimizations

### **1. Update `requirements.txt`**

Create production requirements:
```bash
pip freeze > requirements.txt
```

### **2. Disable Debug Mode**

In `app.py`:
```python
# Change reload=True to reload=False
uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
```

### **3. Use Production ASGI Server**

Install Gunicorn:
```bash
pip install gunicorn
```

Run with Gunicorn:
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **4. Optimize Database**

Use PostgreSQL for production (optional):
```python
# In .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/eva_db
```

---

## 🌍 Domain & DNS Setup

### **1. Get a Domain**
- Namecheap, GoDaddy, or Google Domains
- Example: `eva-teaching.com`

### **2. Point DNS to Server**

**A Record:**
```
Type: A
Name: @
Value: YOUR_SERVER_IP
TTL: 3600
```

**CNAME for www:**
```
Type: CNAME
Name: www
Value: eva-teaching.com
TTL: 3600
```

### **3. Update URLs**

In your frontend HTML files, update:
```javascript
// teacher_dashboard/index.html
const WS_URL = 'wss://eva-teaching.com/ws/teacher';
const API_URL = 'https://eva-teaching.com/api';
```

---

## 📱 Mobile Access

EVA works on mobile browsers! Students can join from:
- 📱 iPhone/iPad (Safari)
- 📱 Android (Chrome)
- 💻 Tablets

**Tip:** Create a QR code for easy student access:
```python
import qrcode
qr = qrcode.make('https://eva-teaching.com/student')
qr.save('student_qr.png')
```

---

## 🐛 Troubleshooting

### **Issue: Models not loading**
```bash
# Check model files exist
ls -lh models/

# Re-download if needed
python models/download_models.py
```

### **Issue: Port already in use**
```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 PID  # Mac/Linux
taskkill /PID PID /F  # Windows
```

### **Issue: Database locked**
```bash
# Stop all EVA processes
# Delete lock files
rm database/eva.db-shm database/eva.db-wal
# Restart
```

### **Issue: High CPU usage**
- Reduce frame rate in student webcam capture
- Use smaller YOLO model (yolov8n instead of yolov8m)
- Limit concurrent sessions

---

## 📈 Scaling

### **For Large Classes (50+ students):**

1. **Use Redis for session state:**
```bash
pip install redis aioredis
```

2. **Load balance with Nginx:**
```nginx
upstream eva_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}
```

3. **Separate database server:**
   - Move SQLite to PostgreSQL
   - Use separate DB server

4. **CDN for static files:**
   - Use Cloudflare or AWS CloudFront
   - Serve HTML/CSS/JS from CDN

---

## ✅ Post-Deployment Checklist

- [ ] Application accessible via domain/IP
- [ ] HTTPS enabled (if public)
- [ ] Environment variables set
- [ ] Database backups configured
- [ ] Monitoring/logging enabled
- [ ] Teacher dashboard working
- [ ] Student link working
- [ ] Webcam permissions working
- [ ] Emotion detection accurate
- [ ] Chat sentiment working
- [ ] Session reports generating
- [ ] Mobile access tested

---

## 🎓 Recommended Setup for Schools

**Small School (1-5 teachers, <100 students):**
- Deploy on Render.com (free tier)
- Use SQLite database
- Share student link via QR code

**Medium School (5-20 teachers, 100-500 students):**
- Deploy on Railway or AWS EC2 (t2.medium)
- Use PostgreSQL database
- Custom domain with HTTPS
- Regular backups

**Large School/University (20+ teachers, 500+ students):**
- Deploy on AWS/GCP with load balancing
- PostgreSQL with replication
- Redis for caching
- CDN for static assets
- Dedicated monitoring (Datadog, New Relic)

---

## 📞 Support

For deployment issues:
1. Check logs first
2. Review this guide
3. Check GitHub issues
4. Contact support

---

## 🎉 You're Ready!

Your EVA application is now deployed and ready to help teachers understand student emotions in real-time!

**Next Steps:**
1. Train teachers on using the dashboard
2. Share student link with students
3. Monitor first few sessions
4. Collect feedback and iterate

Good luck! 🚀
