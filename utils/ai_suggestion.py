"""
utils/ai_suggestion.py
──────────────────────
Feature 1: AI-powered teacher response suggestions.
When a student is confused/frustrated, EVA suggests what the teacher should say.
Uses Groq free API (already configured in .env).
"""
from dotenv import load_dotenv
import os
load_dotenv() 
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SUGGESTION_PROMPT = """You are helping a teacher in an online class.

A student named {name} is feeling {emotion}.
Their last message: "{message}"
Subject being taught: {subject}

Give ONE short, practical suggestion (2-3 sentences max) for what the teacher should say or do RIGHT NOW to help this student.
Be specific, warm, and actionable.
Do NOT start with "I suggest" or "You should". 
Just give the direct suggestion text.
Keep it under 40 words."""

FALLBACK_SUGGESTIONS = {
    "confused": [
        "Pause and ask '{name}, which part is unclear? Let me explain with a simpler example.'",
        "Try explaining the same concept using a real-life analogy that students can relate to.",
        "Break the topic into smaller steps. Ask '{name} to tell you where they got stuck."
    ],
    "frustrated": [
        "Acknowledge their effort: 'This is a tough topic — you're doing well to keep trying.'",
        "Slow down and revisit the last concept. Ask if the class wants a quick recap.",
        "Give a short 2-minute break, then restart with the simplest version of the concept."
    ],
    "disengaged": [
        "Ask {name} a direct but easy question to re-engage them with the topic.",
        "Try adding a quick poll or interactive question to bring energy back to the class.",
        "Share a surprising or interesting fact related to the current topic."
    ],
    "anxious": [
        "Reassure {name}: 'There are no wrong questions here. Take your time.'",
        "Lower the pressure — remind students that confusion is a normal part of learning.",
        "Give a simple practice example that builds confidence step by step."
    ],
    "sad": [
        "Check in with {name} privately after class. A quick message can make a big difference.",
        "Acknowledge that the topic is challenging and praise recent effort.",
        "Share an encouraging word: progress takes time and every step counts."
    ],
}

import random

class AISuggestion:
    def __init__(self):
        self._client = None
        self._model = "llama-3.3-70b-versatile"
        self._ready = False
        self._init()

    def _init(self):
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        api_key  = os.getenv("LLM_API_KEY", "").strip()

        if not api_key:
            logger.warning("⚠️  LLM_API_KEY not set — AI suggestions will use fallback templates")
            return

        try:
            if provider == "groq":
                from groq import Groq
                self._client = Groq(api_key=api_key)
                self._ready = True
                logger.info("✅ AI Suggestion engine ready (Groq)")
            elif provider == "deepseek":
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                self._model  = os.getenv("LLM_MODEL", "deepseek-chat")
                self._ready  = True
                logger.info("✅ AI Suggestion engine ready (DeepSeek)")
            elif provider == "openrouter":
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                self._model  = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout")
                self._ready  = True
                logger.info("✅ AI Suggestion engine ready (OpenRouter)")
        except ImportError as e:
            logger.warning(f"⚠️  AI suggestion provider not installed: {e}")

    def get_suggestion(
        self,
        student_name: str,
        emotion: str,
        message: str,
        subject: str = "general"
    ) -> Dict:
        """
        Generate a teaching suggestion for the teacher.
        Returns dict with 'suggestion' and 'source' (ai or template).
        """
        if self._ready and self._client:
            return self._ai_suggestion(student_name, emotion, message, subject)
        return self._template_suggestion(student_name, emotion)

    def _ai_suggestion(self, name: str, emotion: str, message: str, subject: str) -> Dict:
        try:
            prompt = SUGGESTION_PROMPT.format(
                name=name, emotion=emotion, message=message, subject=subject
            )
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            suggestion = resp.choices[0].message.content.strip()
            return {"suggestion": suggestion, "source": "ai"}
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
            return self._template_suggestion(name, emotion)

    def _template_suggestion(self, name: str, emotion: str) -> Dict:
        templates = FALLBACK_SUGGESTIONS.get(emotion, FALLBACK_SUGGESTIONS["confused"])
        suggestion = random.choice(templates).replace("{name}", name)
        return {"suggestion": suggestion, "source": "template"}

    def is_ready(self) -> bool:
        return self._ready