"""Text Sentiment Analyzer using VADER + keyword rules"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

CONFUSED_KW  = ["confused","don't understand","dont understand","lost","stuck",
                 "not clear","help","explain","huh","what is","i don't get"]
FRUSTRATED_KW= ["frustrated","this is hard","too hard","give up","can't","impossible",
                 "hate this","wrong again","still wrong","doesn't work"]
BORED_KW     = ["boring","bored","already know","too easy","skip","whatever","fine"]
EXCITED_KW   = ["love this","amazing","finally","makes sense","got it","i understand",
                 "cool","great","yes!","awesome"]


# Feature 6: Hindi/multilingual confusion keywords
HINDI_CONFUSED_KW = [
    "samajh nahi aaya", "samajh nahi", "nahi samjha", "kuch samajh nahi",
    "mujhe nahi pata", "explain karo", "dobara batao", "ek baar aur",
    "mushkil hai", "baar baar galat", "nahi ho raha"
]
HINDI_FRUSTRATED_KW = [
    "bahut mushkil", "ye nahi hoga", "chod diya", "bakwaas hai",
    "kuch nahi aata", "galat hai ye"
]
HINDI_HAPPY_KW = [
    "samajh aa gaya", "accha laga", "sahi hai", "bilkul sahi",
    "theek hai", "bahut accha", "got it"
]

class TextSentimentAnalyzer:
    def __init__(self):
        self._ready = False
        self._vader = None
        self._init()

    def _init(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._ready = True
            logger.info("✅ VADER sentiment ready")
        except ImportError:
            logger.warning("⚠️  vaderSentiment not installed")

    def analyze(self, text: str) -> Dict:
        text_lower = text.lower()

        # Feature 6: Hindi/multilingual keyword check first
        if any(k in text_lower for k in HINDI_CONFUSED_KW):
            return self._make("confused", 0.88, text, "hindi_keyword")
        if any(k in text_lower for k in HINDI_FRUSTRATED_KW):
            return self._make("angry", 0.85, text, "hindi_keyword")
        if any(k in text_lower for k in HINDI_HAPPY_KW):
            return self._make("happy", 0.85, text, "hindi_keyword")

        # English keyword rules
        if any(k in text_lower for k in CONFUSED_KW):
            return self._make("confused", 0.88, text, "keyword")
        if any(k in text_lower for k in FRUSTRATED_KW):
            return self._make("angry", 0.85, text, "keyword")
        if any(k in text_lower for k in BORED_KW):
            return self._make("bored", 0.80, text, "keyword")
        if any(k in text_lower for k in EXCITED_KW):
            return self._make("happy", 0.85, text, "keyword")

        # VADER fallback
        if self._vader:
            scores = self._vader.polarity_scores(text)
            c = scores["compound"]
            if c >= 0.5:    return self._make("happy",   min(0.95, c),      text, "vader")
            if c >= 0.05:   return self._make("neutral", 0.6 + c * 0.3,    text, "vader")
            if c <= -0.5:   return self._make("angry",   min(0.95, abs(c)), text, "vader")
            if c <= -0.05:  return self._make("sad",     0.6 + abs(c)*0.3,  text, "vader")
            return self._make("neutral", 0.7, text, "vader")

        return self._make("neutral", 0.5, text, "none")

    def _make(self, emotion: str, confidence: float, text: str, method: str) -> Dict:
        return {"emotion": emotion, "confidence": round(confidence, 3),
                "text": text, "method": method}

    def is_ready(self) -> bool:
        return self._ready