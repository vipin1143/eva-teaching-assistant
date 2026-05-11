@echo off
REM ══════════════════════════════════════════════════════════════
REM EVA Deployment Script for Windows
REM Quick deployment for Windows servers
REM ══════════════════════════════════════════════════════════════

echo 🚀 EVA Deployment Script (Windows)
echo ════════════════════════════════════════════════════════════

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.11+
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Check if models exist
echo 🔍 Checking model files...
if not exist "models\eva_emotion_v1.onnx" (
    echo ⚠️  Warning: eva_emotion_v1.onnx not found
    echo    Download from your training output or use FER+ fallback
)

if not exist "models\emotion_ferplus_8.onnx" (
    echo 📥 Downloading FER+ model...
    python models\download_models.py
)

if not exist "yolov8n-face.pt" (
    echo ⚠️  Warning: yolov8n-face.pt not found
    echo    Will use YOLOv8m (slower) or Haar Cascade fallback
)

REM Check .env file
echo 🔐 Checking environment variables...
if not exist ".env" (
    echo ⚠️  Warning: .env file not found
    echo    Creating template...
    (
        echo # Database
        echo DATABASE_URL=sqlite+aiosqlite:///./database/eva.db
        echo.
        echo # AI/LLM (for suggestions^)
        echo LLM_PROVIDER=groq
        echo LLM_API_KEY=your_groq_api_key_here
        echo LLM_MODEL=llama-3.3-70b-versatile
        echo.
        echo # Zoom Integration (optional^)
        echo ZOOM_WEBHOOK_SECRET=
        echo.
        echo # Google Meet Integration (optional^)
        echo GOOGLE_MEET_WEBHOOK_URL=
    ) > .env
    echo    ✅ Created .env template - please edit with your API keys
)

REM Create database directory
echo 📁 Setting up database...
if not exist "database" mkdir database

REM Test import
echo 🧪 Testing imports...
python -c "import fastapi; import uvicorn; import onnxruntime; import cv2; import mediapipe; print('✅ All core dependencies imported successfully')"
if errorlevel 1 (
    echo ❌ Import test failed
    exit /b 1
)

echo.
echo ════════════════════════════════════════════════════════════
echo ✅ Deployment setup complete!
echo ════════════════════════════════════════════════════════════
echo.
echo 📝 Next steps:
echo    1. Edit .env file with your API keys
echo    2. Run: python app.py
echo    3. Access: http://localhost:8000/teacher
echo.
echo 🌐 For production deployment:
echo    - See DEPLOYMENT_GUIDE.md for details
echo    - Consider using IIS or Windows Service
echo.
pause
