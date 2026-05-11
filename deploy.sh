#!/bin/bash
# ══════════════════════════════════════════════════════════════
# EVA Deployment Script
# Quick deployment for Linux/Mac servers
# ══════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "🚀 EVA Deployment Script"
echo "════════════════════════════════════════════════════════════"

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if models exist
echo "🔍 Checking model files..."
if [ ! -f "models/eva_emotion_v1.onnx" ]; then
    echo "⚠️  Warning: eva_emotion_v1.onnx not found"
    echo "   Download from your training output or use FER+ fallback"
fi

if [ ! -f "models/emotion_ferplus_8.onnx" ]; then
    echo "📥 Downloading FER+ model..."
    python models/download_models.py
fi

if [ ! -f "yolov8n-face.pt" ]; then
    echo "⚠️  Warning: yolov8n-face.pt not found"
    echo "   Will use YOLOv8m (slower) or Haar Cascade fallback"
fi

# Check .env file
echo "🔐 Checking environment variables..."
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Creating template..."
    cat > .env << EOF
# Database
DATABASE_URL=sqlite+aiosqlite:///./database/eva.db

# AI/LLM (for suggestions)
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile

# Zoom Integration (optional)
ZOOM_WEBHOOK_SECRET=

# Google Meet Integration (optional)
GOOGLE_MEET_WEBHOOK_URL=
EOF
    echo "   ✅ Created .env template - please edit with your API keys"
fi

# Create database directory
echo "📁 Setting up database..."
mkdir -p database

# Test import
echo "🧪 Testing imports..."
python3 -c "
import fastapi
import uvicorn
import onnxruntime
import cv2
import mediapipe
print('✅ All core dependencies imported successfully')
" || { echo "❌ Import test failed"; exit 1; }

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Deployment setup complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Run: python app.py"
echo "   3. Access: http://localhost:8000/teacher"
echo ""
echo "🌐 For production deployment:"
echo "   - Use gunicorn: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker"
echo "   - Set up Nginx reverse proxy"
echo "   - Enable HTTPS with Let's Encrypt"
echo "   - See DEPLOYMENT_GUIDE.md for details"
echo ""
