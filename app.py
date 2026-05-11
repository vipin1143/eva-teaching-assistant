"""
EVA – Emotion-Adaptive Virtual Teaching Assistant
Teacher-only architecture with hybrid detection
Run: uvicorn app:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()
import os
import uvicorn
from backend.server import create_app

app = create_app()

if __name__ == "__main__":
    # Get port from environment variable (for Render/Heroku) or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*60)
    print("🎓  EVA – Emotion-Adaptive Teaching Assistant")
    print("="*60)
    print(f"📡  Backend API   : http://0.0.0.0:{port}")
    print(f"🖥️   Teacher Dashboard: http://0.0.0.0:{port}/teacher")
    print(f"🔗  Student Link  : http://0.0.0.0:{port}/student")
    print(f"📖  API Docs      : http://0.0.0.0:{port}/docs")
    print("="*60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)