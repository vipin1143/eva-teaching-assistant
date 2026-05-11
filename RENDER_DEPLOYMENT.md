# 🚀 Render.com Deployment Guide - Step by Step

## Prerequisites

Before starting, make sure you have:
- ✅ A GitHub account (free)
- ✅ A Render.com account (free) - Sign up at https://render.com
- ✅ Your Groq API key (free) - Get from https://console.groq.com

---

## 📋 Step-by-Step Deployment

### **Step 1: Prepare Your Code**

1. **Open terminal in your eva-project folder**

2. **Initialize Git repository:**
```bash
git init
```

3. **Add all files:**
```bash
git add .
```

4. **Commit your code:**
```bash
git commit -m "Initial commit - EVA v3.3 production ready"
```

---

### **Step 2: Create GitHub Repository**

1. **Go to GitHub:** https://github.com

2. **Click the "+" icon** (top right) → **"New repository"**

3. **Fill in details:**
   - Repository name: `eva-teaching-assistant` (or any name you like)
   - Description: `EVA - Emotion-Adaptive Virtual Teaching Assistant`
   - Visibility: **Public** (required for Render free tier)
   - ❌ **DO NOT** initialize with README (we already have code)

4. **Click "Create repository"**

5. **Copy the commands shown** and run in your terminal:
```bash
git remote add origin https://github.com/YOUR_USERNAME/eva-teaching-assistant.git
git branch -M main
git push -u origin main
```

**✅ Your code is now on GitHub!**

---

### **Step 3: Deploy on Render**

1. **Go to Render:** https://render.com

2. **Sign up / Log in:**
   - Click "Get Started" or "Sign In"
   - Sign in with GitHub (recommended)
   - Authorize Render to access your repositories

3. **Create New Web Service:**
   - Click **"New +"** (top right)
   - Select **"Web Service"**

4. **Connect Repository:**
   - Find your `eva-teaching-assistant` repository
   - Click **"Connect"**
   - If you don't see it, click "Configure account" and grant access

5. **Configure Service:**

   **Name:** `eva-teaching-assistant` (or your preferred name)
   
   **Region:** Choose closest to you (e.g., Oregon, Frankfurt)
   
   **Branch:** `main`
   
   **Root Directory:** Leave empty
   
   **Runtime:** `Python 3`
   
   **Build Command:**
   ```
   pip install -r requirements.txt
   ```
   
   **Start Command:**
   ```
   python app.py
   ```
   
   **Plan:** Select **"Free"** (0$/month)

6. **Add Environment Variables:**
   
   Scroll down to **"Environment Variables"** section
   
   Click **"Add Environment Variable"** and add these:

   | Key | Value |
   |-----|-------|
   | `PORT` | `8000` |
   | `PYTHON_VERSION` | `3.11.0` |
   | `DATABASE_URL` | `sqlite+aiosqlite:///./database/eva.db` |
   | `LLM_PROVIDER` | `groq` |
   | `LLM_MODEL` | `llama-3.3-70b-versatile` |
   | `LLM_API_KEY` | `YOUR_GROQ_API_KEY` ⚠️ |

   **⚠️ IMPORTANT:** Replace `YOUR_GROQ_API_KEY` with your actual Groq API key from https://console.groq.com

7. **Click "Create Web Service"**

---

### **Step 4: Wait for Deployment**

1. **Render will now:**
   - Clone your repository
   - Install dependencies (this takes 5-10 minutes)
   - Download model files
   - Start your application

2. **Watch the logs:**
   - You'll see real-time deployment logs
   - Look for these success messages:
     ```
     ✅ EVA custom emotion model loaded
     ✅ YOLOv8n-FACE detector ready
     ✅ Database ready
     ✅ All components loaded
     ```

3. **Wait for "Live" status:**
   - When deployment succeeds, you'll see a green **"Live"** badge
   - Your app URL will be shown at the top (e.g., `https://eva-teaching-assistant.onrender.com`)

---

### **Step 5: Access Your Application**

Once deployment is complete:

1. **Copy your Render URL** (e.g., `https://eva-teaching-assistant.onrender.com`)

2. **Access the dashboards:**
   - **Teacher Dashboard:** `https://your-app.onrender.com/teacher`
   - **Student Link:** `https://your-app.onrender.com/student`
   - **API Docs:** `https://your-app.onrender.com/docs`

3. **Test it:**
   - Open teacher dashboard
   - Start a session
   - Open student link in another tab/device
   - Allow webcam access
   - Smile and check if emotion is detected! 😊

---

## 🎉 You're Live!

Your EVA application is now deployed and accessible worldwide!

**Share the student link with your students:**
```
https://your-app.onrender.com/student
```

---

## ⚠️ Important Notes

### **Free Tier Limitations:**

1. **Cold Starts:**
   - App sleeps after 15 minutes of inactivity
   - First request after sleep takes 30-60 seconds to wake up
   - Subsequent requests are fast

2. **Monthly Limits:**
   - 750 hours/month (enough for continuous use)
   - Resets on 1st of each month

3. **Storage:**
   - Database resets on each deployment
   - For persistent data, upgrade to paid plan or use external database

### **Solutions:**

**To prevent cold starts:**
- Use a service like UptimeRobot (free) to ping your app every 10 minutes
- Upgrade to paid plan ($7/month) for always-on service

**For persistent database:**
- Use external PostgreSQL (Render provides free PostgreSQL)
- Or upgrade to paid plan with persistent disk

---

## 🔧 Troubleshooting

### **Issue: Deployment Failed**

**Check logs for errors:**
1. Click on your service in Render dashboard
2. Go to "Logs" tab
3. Look for error messages

**Common issues:**

**1. Missing dependencies:**
```
Error: No module named 'xyz'
```
**Solution:** Add missing package to `requirements.txt`

**2. Model files too large:**
```
Error: File too large for Git
```
**Solution:** Models should auto-download. If not, check `models/download_models.py`

**3. Port binding error:**
```
Error: Address already in use
```
**Solution:** Make sure `PORT` environment variable is set to `8000`

---

### **Issue: App is Slow**

**Cause:** Cold start (app was sleeping)

**Solutions:**
1. Wait 30-60 seconds for first request
2. Set up UptimeRobot to keep app awake
3. Upgrade to paid plan

---

### **Issue: Webcam Not Working**

**Cause:** HTTPS required for webcam access

**Solution:** Render provides HTTPS by default, so this should work. If not:
1. Make sure you're using `https://` not `http://`
2. Check browser permissions
3. Try different browser (Chrome recommended)

---

### **Issue: Database Resets**

**Cause:** Free tier doesn't have persistent storage

**Solutions:**
1. Use external PostgreSQL database (Render provides free PostgreSQL)
2. Upgrade to paid plan with persistent disk
3. Accept that sessions reset on each deployment (OK for testing)

---

## 🔄 Updating Your App

When you make changes:

1. **Commit changes:**
```bash
git add .
git commit -m "Updated emotion detection"
git push origin main
```

2. **Render auto-deploys:**
   - Render detects the push
   - Automatically rebuilds and redeploys
   - Takes 5-10 minutes

3. **Manual deploy:**
   - Go to Render dashboard
   - Click "Manual Deploy" → "Deploy latest commit"

---

## 📊 Monitoring

### **View Logs:**
1. Go to Render dashboard
2. Click your service
3. Click "Logs" tab
4. See real-time logs

### **Check Metrics:**
1. Click "Metrics" tab
2. See CPU, memory, bandwidth usage

### **Set up Alerts:**
1. Click "Settings"
2. Add notification email
3. Get alerts for downtime

---

## 💰 Upgrading (Optional)

If you need more:

**Starter Plan ($7/month):**
- No cold starts (always on)
- 512 MB RAM
- Persistent disk
- Custom domain

**Standard Plan ($25/month):**
- 2 GB RAM
- Better performance
- Priority support

---

## 🎓 Next Steps

1. ✅ **Test thoroughly** with real students
2. ✅ **Share student link** via QR code or URL
3. ✅ **Monitor logs** for any issues
4. ✅ **Collect feedback** from teachers
5. ✅ **Set up UptimeRobot** to prevent cold starts (optional)

---

## 📞 Need Help?

**Render Support:**
- Docs: https://render.com/docs
- Community: https://community.render.com
- Status: https://status.render.com

**EVA Support:**
- Check `DEPLOYMENT_GUIDE.md`
- Review logs for errors
- Test locally first

---

## ✅ Deployment Checklist

- [x] Code pushed to GitHub
- [x] Render service created
- [x] Environment variables set
- [x] Deployment successful
- [x] App is live
- [x] Teacher dashboard accessible
- [x] Student link working
- [x] Webcam access working
- [x] Emotion detection working
- [ ] UptimeRobot configured (optional)
- [ ] Custom domain added (optional)

---

**Congratulations! Your EVA app is now live on Render! 🎉**
