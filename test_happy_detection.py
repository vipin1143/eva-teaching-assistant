"""
Test script to verify happy emotion detection
Run this to check if happy faces are being detected correctly
"""

import sys
import os
import logging

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from emotion_detector import EmotionDetector

def test_happy_detection():
    """Test the emotion detector with a sample image"""
    
    print("=" * 60)
    print("🧪 HAPPY EMOTION DETECTION TEST")
    print("=" * 60)
    
    # Initialize detector
    print("\n1️⃣ Initializing emotion detector...")
    detector = EmotionDetector()
    
    # Check status
    status = detector.is_ready()
    print("\n2️⃣ Detector Status:")
    for component, ready in status.items():
        icon = "✅" if ready else "❌"
        print(f"   {icon} {component}: {ready}")
    
    print("\n3️⃣ Primary Model:", status.get('primary_model', 'unknown'))
    
    # Instructions for testing
    print("\n" + "=" * 60)
    print("📸 TO TEST WITH YOUR OWN IMAGE:")
    print("=" * 60)
    print("""
1. Take a photo with a happy/smiling face
2. Convert it to base64 (or use the web interface)
3. Call: detector.detect_from_base64(image_base64)

Example code:
    import base64
    with open('happy_face.jpg', 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    result = detector.detect_from_base64(img_b64)
    print(f"Detected: {result['emotion']} ({result['confidence']*100:.1f}%)")
    print(f"All emotions: {result.get('all_emotions', {})}")

🔍 Check the logs above for:
   - "🎭 EVA model: happy" messages
   - Top 3 predictions with percentages
   - Smoothing decisions (if using smooth_key)
""")
    
    print("\n" + "=" * 60)
    print("💡 CALIBRATION SETTINGS (v3.1):")
    print("=" * 60)
    print("""
   happy:    2.20 (BOOSTED for better detection)
   neutral:  1.15 (reduced to favor happy)
   disgust:  0.40 (suppressed - often confused with happy)
   fear:     0.50 (suppressed - FER-2013 noise)
   surprise: 1.00 (neutral)
   angry:    1.00 (neutral)
   sad:      0.90 (slightly reduced)
   
   Uncertainty threshold: 0.18 (lowered from 0.22)
   Smoothing window: 3 frames
""")
    
    print("\n✅ Test setup complete!")
    print("   Run your application and check the logs for emotion predictions.\n")

if __name__ == "__main__":
    test_happy_detection()
